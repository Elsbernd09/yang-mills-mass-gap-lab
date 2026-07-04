from ymlab.lattice import Lattice
from ymlab.su2 import identity, random_su2, is_su2


def test_lattice_site_count():
    lattice = Lattice(shape=(4, 5), cold_start=True)

    assert lattice.number_of_sites() == 20


def test_lattice_link_count():
    lattice = Lattice(shape=(4, 5), cold_start=True)

    assert lattice.number_of_links() == 40


def test_periodic_shift_forward():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    assert lattice.shift((3, 1), direction=0, amount=1) == (0, 1)


def test_periodic_shift_backward():
    lattice = Lattice(shape=(4, 4), cold_start=True)

    assert lattice.shift((0, 2), direction=0, amount=-1) == (3, 2)


def test_cold_start_links_are_identity():
    lattice = Lattice(shape=(2, 2), cold_start=True)

    for site in lattice.sites():
        for mu in range(lattice.dim):
            assert (lattice.get_link(site, mu) == identity()).all()


def test_set_link_accepts_su2():
    lattice = Lattice(shape=(2, 2), cold_start=True)
    u = random_su2()

    lattice.set_link((0, 0), 0, u)

    assert is_su2(lattice.get_link((0, 0), 0))
