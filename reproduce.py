#!/usr/bin/env python3
"""Compute Yukawa reference values, constants and oscillator comparisons.
Basis and mesh differences assess convergence; they are not continuum error bounds.
Run with Python 3; numbers.json is written beside this script.
"""
from pathlib import Path
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP
import json
import math
import platform

import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal

mp.mp.dps = 50


def slater_ritz(delta, size, zmin, zmax):
    """Ground s state in u_j(r)=r exp(-z_j r), normalized with measure dr."""
    zmin, zmax = mp.mpf(zmin), mp.mpf(zmax)
    z = [zmin * (zmax / zmin) ** (mp.mpf(i) / (size - 1)) for i in range(size)]
    S, H, R = (mp.matrix(size, size) for _ in range(3))
    for i in range(size):
        for j in range(i, size):
            s = z[i] + z[j]
            S[i, j] = S[j, i] = 2 / s**3
            H[i, j] = H[j, i] = z[i] * z[j] / s**3 - 1 / (s + delta)**2
            R[i, j] = R[j, i] = 6 / s**4
    inverse_L = mp.inverse(mp.cholesky(S))
    A = inverse_L * H * inverse_L.T
    E, Q = mp.eigsy((A + A.T) / 2)
    c = inverse_L.T * Q[:, 0]
    norm = (c.T * S * c)[0]
    radius = (c.T * R * c)[0] / norm
    contact = mp.fsum(c[i] / mp.sqrt(norm) for i in range(size))**2 / (4 * mp.pi)
    return E[0], radius, contact


def finite_difference(delta, h, radius=60.0):
    """Radial Dirichlet discretization of -u''/2-exp(-delta*r)u/r."""
    size = int(round(radius / h)) - 1
    r = h * np.arange(1, size + 1)
    E, v = eigh_tridiagonal(1 / h**2 - np.exp(-delta * r) / r,
        -0.5 / h**2 * np.ones(size - 1), select='i', select_range=(0, 0))
    u = v[:, 0] / math.sqrt(h)
    # Five-point forward derivative at r=0, where u(0)=0.
    slope = (48*u[0] - 36*u[1] + 16*u[2] - 3*u[3]) / (12*h)
    return float(E[0]), float(np.sum(r * v[:, 0]**2)), float(slope**2 / (4*np.pi))


def hulthen_mean(mu):
    return 1 / (2 - mu) + mp.mpf('0.5') + 1 / (2 + mu)


def hulthen_contact(mu):
    return (4 - mu**2) / (4 * mp.pi)


def hulthen_minmax(n, mu):
    # A missing discrete s level contributes the continuum threshold zero.
    return -(1 - n*n*mu/2)**2 / (2*n*n) if n*n*mu < 2 else mp.mpf(0)


def formatted(value, places, lower=False):
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-places),
        rounding=ROUND_FLOOR if lower else ROUND_HALF_UP))


def main():
    checks, rows = [], []

    def check(name, passed, detail):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})
        if not passed:
            raise AssertionError(name + ': ' + str(detail))

    f = lambda x: mp.exp(-x) * (1/x - 1/mp.sinh(x))
    xc = mp.findroot(lambda x: mp.diff(f, x), (mp.mpf('.6'), mp.mpf('1.1')))
    c = f(xc)
    a = mp.findroot(lambda x: (2-x)*mp.exp(x)-2, (mp.mpf('1.4'), mp.mpf('1.8')))
    b = 1 / (mp.e * (2-a))
    check('printed Hall constant', formatted(c, 6, True) == '0.055810', str(c))
    check('printed binding constant', formatted(2*b/a, 5, True) == '1.13611', str(2*b/a))
    expected = [
        ['1.5025', '1.5133', '1.5152', '0.3135', '0.3140', '0.3175', '0.984'],
        ['1.5101', '1.5497', '1.5638', '0.2992', '0.3027', '0.3151', '0.964'],
        ['1.5667', '1.8066', '2.1000', '0.1989', '0.2390', '0.2984', '0.776']]
    for ds, printed in zip(('0.1', '0.2', '0.5'), expected):
        d = mp.mpf(ds)
        mu = mp.sqrt(6) * d
        _, ra, pa = slater_ritz(d, 40, '0.01', '60')
        eb, rb, pb = slater_ritz(d, 50, '0.005', '100')
        gap = hulthen_minmax(2, 2*d) - hulthen_minmax(1, 2*d)
        lower_bound = 1 - c*d/gap
        values = [hulthen_mean(d), rb, hulthen_mean(mu), hulthen_contact(mu), pb,
                  hulthen_contact(d), lower_bound]
        differences = [mp.mpf(0), abs(ra-rb), mp.mpf(0), mp.mpf(0), abs(pa-pb),
                       mp.mpf(0), mp.mpf(0)]
        for j, (value, difference, shown) in enumerate(zip(values, differences, printed)):
            check('table delta=' + ds + ' row=' + str(j+1),
                all(formatted(value+t*difference, 3 if j == 6 else 4, j == 6) == shown
                    for t in (-1, 0, 1)),
                {'value': str(value), 'basis_difference': str(difference),
                 'printed': shown, 'rounding': 'down' if j == 6 else 'nearest'})
        meshes = [finite_difference(float(d), h) for h in (.01, .005, .0025)]
        energies, radii, contacts = np.array(meshes).T
        rich = (4*energies[1:] - energies[:-1]) / 3
        fd_energy = (16*rich[1] - rich[0]) / 15
        fd_radius = (4*radii[2] - radii[1]) / 3
        fd_contact = (4*contacts[2] - contacts[1]) / 3
        for name, value, reference, tolerance in (
            ('radial energy', fd_energy, eb, 2e-9),
            ('radius', fd_radius, rb, 2e-7),
            ('contact', fd_contact, pb, 2e-6)):
            difference = abs(value - float(reference))
            check('independent ' + name + ' delta=' + ds, difference < tolerance, difference)
        check('analytic radius ordering delta=' + ds,
              values[0] <= values[1] <= values[2], list(map(str, values[:3])))
        check('analytic contact ordering delta=' + ds,
              values[3] <= values[4] <= values[5], list(map(str, values[3:6])))
        rows.append({'delta': ds, 'values': list(map(str, values)),
            'basis_differences': list(map(str, differences)), 'ritz_energy': str(eb),
            'finite_difference_meshes': meshes, 'richardson': [fd_energy, fd_radius, fd_contact]})

    def spectrum(wx, wy):
        return sorted(wx*(nx+.5) + wy*(ny+.5)
                      for nx in range(12) for ny in range(12))

    w, v = math.sqrt(1.6), math.sqrt(.4)
    for panel, frequencies, expected_third in (
        ('a', [(1, 1), (w, 1), (w, w)], [3, (1+3*w)/2, 3*w]),
        ('b', [(v, 1), (1, 1), (w, 1)], [(3+v)/2, 3, (1+3*w)/2])):
        energies = [spectrum(wx, wy) for wx, wy in frequencies]
        third = [sorted(set(round(e, 11) for e in column))[2] for column in energies]
        # The first omitted oscillator state lies above all 20 values checked.
        complete = all(column[19] < 12*min(wx, wy)+(wx+wy)/2
                       for column, (wx, wy) in zip(energies, frequencies))
        ordered = all(energies[0][k] <= energies[1][k]+1e-10
                      and energies[1][k] <= energies[2][k]+1e-10 for k in range(20))
        reversed_rank = third[1] < third[0] if panel == 'a' else third[1] > third[2]
        check('oscillator panel ' + panel,
            complete and ordered and reversed_rank
            and all(abs(x-y) < 1e-10 for x, y in zip(third, expected_third)),
            {'third_distinct': third, 'expected_third': expected_third,
             'ordered_minmax_values_checked': 20, 'enumeration_complete': complete})
    output = {'versions': {'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'mpmath': mp.__version__},
        'error_scope': 'basis and mesh convergence diagnostics; no interval enclosure of the continuum spectrum',
        'constants': {'Hall_c': str(c), 'Hall_x': str(xc), 'm_star': str(a),
                      'b_star': str(b), 'binding_bound': str(2*b/a)},
        'table': rows, 'checks': checks, 'passed': all(item['passed'] for item in checks)}
    Path(__file__).with_name('numbers.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f"{len(checks)} checks passed; table and constants written to numbers.json")


if __name__ == '__main__':
    main()
