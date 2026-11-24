"""One of each of the later wave, algebra and calculus objects.

Left to right, top row: the Cayley graph of D4 (r turns the rings, s jumps between them), its
table sorted by the cosets of the rotations (a normal subgroup: the quotient is the 2x2 block
pattern), and the homomorphism Z12 -> Z4 (kernel {0, 4, 8} highlighted). Middle: separation of
variables for a heating rod (each mode fades at its own rate, the high ones first), insulated
ends against held ones, and epsilon-delta for sin(x)/x. Bottom: the subgroup lattice of D4, the
Galois correspondence for Q(sqrt 2, sqrt 3), a permutation's tokens travelling, and Green's
theorem around a circle in a vector field.

Run:  uv run python examples/more_wave2.py      (prints what the inspector reads)
"""

from phasor import Scene, algebra, calc, expr, pde

s = Scene("more_wave2", look="calm")
s.set("camera.center", (0.0, 0.0, 0.0))
s.set("camera.height", 21)

s.add(algebra.CayleyGraph(id="graph", group="D4", generators="r s", pos=(-13, 6.5, 0), r=2.2))
s.add(algebra.CayleyTable(id="table", group="D4", subgroup="r", cosets=expr("min(t / 2, 1)"),
                          pos=(-2.0, 6.0, 0), cell=0.75))  # fmt: skip
s.add(algebra.Homomorphism(id="hom", n=12, m=4, a=1, pos=(9.5, 6.5, 0), r=1.6, gap=6.0,
                           progress=expr("min(t / 3, 1)")))  # fmt: skip

s.add(pde.Modes(id="modes", count=9, length=1.0, diffusivity=1.0, time=expr("t / 400"),
                pos=(-17, -1.5, 0), extent=(9, 3.5)))  # fmt: skip
s.add(pde.Profile(id="held", kind="heat", modes=(), time=expr("t / 300"), pos=(-5.5, 0.5, 0),
                  extent=(7, 2.0), exact=0.0, style={"color": "primary"}))  # fmt: skip
s.add(pde.Profile(id="insulated", kind="heat", modes=(), boundary="neumann", method="spectral",
                  time=expr("t / 300"), pos=(-5.5, -2.5, 0), extent=(7, 2.0),
                  style={"color": "primary"}))  # fmt: skip
s.add(calc.Limit(id="limit", function="sin(x)/x", a=0.0, epsilon=expr("0.4 / (1 + t)"),
                 xmin=-7.0, xmax=7.0, pos=(10.5, -1.5, 0)))  # fmt: skip

s.add(algebra.SubgroupLattice(id="lattice", group="D4", pos=(-13.6, -7.2, 0), width=6.6,
                              height=3.6))  # fmt: skip
s.add(algebra.GaloisDiagram(id="galois", primes=(2, 3), pos=(0.5, -6.8, 0), width=4.2,
                            height=3.2))  # fmt: skip
s.add(algebra.Permutation(id="perm", perm=(1, 2, 0, 4, 3), pos=(9.0, -4.4, 0),
                          progress=expr("min(t / 3, 1)")))  # fmt: skip
s.add(calc.VectorField(id="field", p="-y", q="x + 0.3 * y", pos=(14.0, -6.6, 0),
                       extent=(5, 4.6), spacing=0.7))  # fmt: skip
s.add(calc.LoopIntegral(id="green", p="-y", q="x + 0.3 * y", measure="circulation",
                        radius=1.6, pos=(14.0, -6.6, 0), progress=expr("min(t / 4, 1)"),
                        style={"color": "primary"}))  # fmt: skip

s.marker("end", 4.0)

if __name__ == "__main__":
    e = s.engine

    def info(oid, t=4.0):
        return e.inspect(t, id=oid)["objects"][0]["semantic"]

    print("quotient D4/<r>:", info("table")["quotient"])
    print("Z12 -> Z4 kernel:", info("hom")["kernel"])
    print("mode amplitudes at t=4:", info("modes")["amplitudes"][:4])
    print("heat held / insulated:", info("held")["total_heat"], info("insulated")["total_heat"])
    print("limit:", {k: info("limit")[k] for k in ("epsilon", "delta", "holds")})
    print("Galois pairs:", [(p["field"], p["subgroup_order"]) for p in info("galois")["pairs"]])
    print("permutation:", info("perm")["cycles"], "order", info("perm")["order"])
    print("Green:", info("green")["total"], "=", info("green")["area_integral"])
