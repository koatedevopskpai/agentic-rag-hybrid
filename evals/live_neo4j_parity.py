"""H03 - live Neo4j parity harness.

Compares the deterministic in-memory engine against the same traversal executed
against a real Neo4j instance. The point is to find where the two disagree, not to
confirm they agree on an easy case.

Requires a running Neo4j and:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

If NEO4J_URI is not set the harness prints SKIP and exits 0, so CI (which has no
database) is unaffected.

    NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j \
    NEO4J_PASSWORD=... python evals/live_neo4j_parity.py

Exit 0 when every probe matches, 1 otherwise.
"""

from __future__ import annotations

import os
import sys

from agentic_rag_hybrid import KnowledgeGraph
from agentic_rag_hybrid.neo4j_store import Neo4jGraphStore

# A probe graph deliberately harder than the golden fixture:
#   - diamonds: d is 1 hop from seed AND 2 hops via b -> tests min-hop dedup
#   - a cycle (x -> y -> z -> x plus z -> x reverse) -> tests cycle safety
#   - an alias entity to test seed extraction
#   - an isolated node that must never be returned
PARITY_ENTITIES = [
    ("seed", "root", []),
    ("b", "child", []),
    ("c", "child", []),
    ("d", "grandchild", []),   # 1 hop direct, and 2 hops via b
    ("e", "child", []),
    ("x", "cycle", []),
    ("y", "cycle", []),
    ("z", "cycle", []),
    ("aliased", "project", ["Alias Target"]),
    ("lonely", "isolated", []),
]

PARITY_RELATIONS = [
    ("seed", "LINKS", "b"),
    ("seed", "LINKS", "c"),
    ("seed", "LINKS", "d"),    # d at hop 1
    ("b", "LINKS", "d"),       # d again at hop 2 -> diamond
    ("c", "LINKS", "e"),
    ("seed", "LINKS", "x"),
    ("x", "LINKS", "y"),
    ("y", "LINKS", "z"),
    ("z", "LINKS", "x"),       # back edge: x reachable at hop 2 as well as hop 0
    ("seed", "LINKS", "aliased"),
]


def build_memory_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    for eid, etype, aliases in PARITY_ENTITIES:
        g.add_entity(eid, type=etype, aliases=tuple(aliases))
    for src, rel, dst in PARITY_RELATIONS:
        g.add_relation(src, rel, dst)
    return g


def seed_neo4j(store: Neo4jGraphStore) -> None:
    with store._driver.session() as session:
        session.run("MATCH (n:Node) DETACH DELETE n")
        session.run("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
        for eid, etype, aliases in PARITY_ENTITIES:
            session.run(
                "MERGE (n:Node {id: $id}) SET n.type = $type, n.aliases = $aliases",
                id=eid,
                type=etype,
                aliases=aliases,
            )
        for src, rel, dst in PARITY_RELATIONS:
            session.run(
                f"MATCH (a:Node {{id: $src}}), (b:Node {{id: $dst}}) MERGE (a)-[:{rel}]->(b)",
                src=src,
                dst=dst,
            )


def shape(rows: list[dict]) -> list[tuple[str, int]]:
    return [(r["node"], r["hops"]) for r in rows]


def main() -> int:
    if not os.getenv("NEO4J_URI"):
        print("SKIP: NEO4J_URI not set - live parity harness not run.")
        return 0

    mem = build_memory_graph()
    store = Neo4jGraphStore()
    probes: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        probes.append((name, ok, detail))

    try:
        seed_neo4j(store)

        # 1. Seeds appear once, at hop 0, in both backends.
        for label, rows in (("memory", mem.traverse(["seed"], 2)), ("live", store.traverse(["seed"], 2))):
            hops_for_seed = [h for n, h in shape(rows) if n == "seed"]
            record(f"seed at hop 0 ({label})", hops_for_seed == [0], f"seed hops={hops_for_seed}")
            record(
                f"no duplicate nodes ({label})",
                len(rows) == len({n for n, _ in shape(rows)}),
                f"{len(rows)} rows for {len({n for n, _ in shape(rows)})} distinct nodes",
            )

        # 2. Node set matches at each hop bound.
        for hops in (1, 2):
            ms = dict(shape(mem.traverse(["seed"], hops)))
            ls = dict(shape(store.traverse(["seed"], hops)))
            record(f"node set matches at max_hops={hops}", set(ms) == set(ls), f"mem={sorted(ms)} live={sorted(ls)}")

        # 3. Minimum hops per node, the diamond case: d must be 1 in both.
        ms = dict(shape(mem.traverse(["seed"], 2)))
        ls = dict(shape(store.traverse(["seed"], 2)))
        record(
            "min hops per node matches (diamond)",
            ms == ls and ms.get("d") == 1,
            f"mem d={ms.get('d')} live d={ls.get('d')}; expected 1",
        )

        # 4. Cycle safety: x is a seed at 0 and reachable at 2 via z -> x.
        for label, rows in (("memory", mem.traverse(["x"], 2)), ("live", store.traverse(["x"], 2))):
            nodes = [n for n, _ in shape(rows)]
            record(
                f"cycle safe: terminates and dedupes ({label})",
                len(nodes) == len(set(nodes)),
                f"{len(nodes)} rows for {len(set(nodes))} distinct nodes over x->y->z->x",
            )

        # 5. Direction is undirected.
        record(
            "undirected traversal",
            dict(shape(mem.traverse(["z"], 2))).get("seed") is not None
            and dict(shape(store.traverse(["z"], 2))).get("seed") is not None,
            "seed reachable from z in both backends",
        )

        # 6. Scores are 1/(1+hops).
        record(
            "score = 1/(1+hops)",
            all(r["score"] == round(1 / (1 + r["hops"]), 4) for r in store.traverse(["seed"], 2)),
            "all live rows",
        )

        # 7. Alias extraction feeds both backends identically. (Extraction is an
        # in-memory concern; Neo4j receives the seeds. So the meaningful check is
        # that both backends traverse from the extracted seeds to the same shape.)
        seeds = mem.extract_entities("what is the Alias Target")
        record(
            "alias extraction finds the aliased entity",
            "aliased" in seeds,
            f"extracted seeds={seeds}",
        )
        record(
            "parity from extracted seeds",
            shape(mem.traverse(seeds, 2)) == shape(store.traverse(seeds, 2)),
            f"seeds={seeds}",
        )

        # 8. Isolated node is never returned.
        record(
            "isolated node excluded",
            "lonely" not in dict(shape(store.traverse(["seed"], 2))),
            "lonely not reachable from seed",
        )

        # 9. Exact ordered sequence parity.
        record(
            "ordered result parity",
            shape(mem.traverse(["seed"], 2)) == shape(store.traverse(["seed"], 2)),
            "mem sequence == live sequence",
        )

        print("H03 live Neo4j parity")
        print("-" * 72)
        failures = 0
        for name, ok, detail in probes:
            if not ok:
                failures += 1
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
            if not ok:
                print(f"        {detail}")
        print("-" * 72)
        print(f"{len(probes) - failures}/{len(probes)} probes passed")
        return 1 if failures else 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
