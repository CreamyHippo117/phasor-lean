"""Echoes of a feedback-delay network as a lattice.

With K feedback loops, each echo is labelled by how many laps it took round each loop: a
point n of N^K, arriving at sum n_i d_i. Three loops are a cubic grid you can draw as it is:
the impulse leaves the corner and the echoes flash on, one shell at a time, as the arrival
time sweeps past them (the equal-time surface is the plane sum n_i d_i = t). Then the star
gains arms, one loop after another, up to sixteen: each new loop's echoes slide out of the
old ones, and the frame fills with light. The count of echoes by time t grows like t^K, so
sixteen loops outrun a room's t^3 almost at once. The camera circles the star.

Run:  uv run python examples/echo_lattice.py      (prints how many echoes each stage shows)
"""

from phasor import Scene, nd

s = Scene("echo_lattice", look="prism")
s.use_camera("orbit", azimuth=-40, elevation=22, distance=9.0, target=(0.9, 0.8, 0.9))
s.add(nd.StarLattice(id="echoes", loops=16, max_time=6.5, spacing=0.55, arms=3.0,
                     time=-0.2, point_size=0.0035, afterglow=0.5, color_by="arm"))  # fmt: skip

s.marker("start", 0.0)
s.marker("sweep", 0.5)  # three loops: the grid lights up shell by shell
s.marker("more_loops", 5.0)  # the star gains arms, three to sixteen
s.marker("end", 12.0)
s.at("sweep").to("echoes.time", 6.5, duration=4.0, ease="linear")
s.at("more_loops").to("echoes.arms", 16.0, duration=6.0)
s.at("start").orbit(60.0, until="end")

if __name__ == "__main__":
    e = s.engine
    for t in (0.5, 2.5, 4.8, 8.0, 12.0):
        sem = e.inspect(t, id="echoes")["objects"][0]["semantic"]
        print(f"t={t:g}: {sem['arms']:g} loops, {sem['heard']} echoes heard of "
              f"{sem['shown']} shown (complete to {sem['complete_until']:g} s)")  # fmt: skip
    for issue in e.lint():
        print("lint:", issue)
