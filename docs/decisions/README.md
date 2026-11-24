# Design decisions

Records of the bigger design choices: the problem, the options weighed and why one was
picked. They were written while phasor was being built, so they mention its build stages
("Milestone 2") and an original design document ("SPEC") that is not part of this
repository. The contracts they refer to by number are the seven in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

| Record | Decision |
| --- | --- |
| [0001](0001-layers.md) | The layer order, and how lower layers get render output without importing `render` |
| [0002](0002-layout-bake.md) | What the layout bake depends on; constraints live in the overrides file |
| [0003](0003-bake-for-workers.md) | Parallel render workers apply their parent engine's bake |
| [0004](0004-tweaks-in-the-script.md) | A script scene keeps its tweaks inside the script (accepted, not built yet) |
| [0005](0005-open-object-contract.md) | An open object contract: scenes can define their own types |
