import math


def render(width, height, spheres, light, plane_y=None, plane_albedo=0.7,
           fov_deg=60.0, ambient=0.1, background=0.2):
    t = math.tan(math.radians(fov_deg) / 2)
    aspect = width / height
    lx, ly, lz = light

    def hit_sphere(o, d, s):
        cx, cy, cz, r, _ = s
        oc = (o[0] - cx, o[1] - cy, o[2] - cz)
        b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2]
        c = oc[0] ** 2 + oc[1] ** 2 + oc[2] ** 2 - r * r
        disc = b * b - c
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        for tt in (-b - sq, -b + sq):
            if tt > 1e-9:
                return tt
        return None

    rows = []
    for j in range(height):
        row = []
        py = (1 - 2 * (j + 0.5) / height) * t
        for i in range(width):
            px = (2 * (i + 0.5) / width - 1) * aspect * t
            n = math.sqrt(px * px + py * py + 1)
            d = (px / n, py / n, 1 / n)
            best, kind = None, None
            for s in spheres:
                tt = hit_sphere((0, 0, 0), d, s)
                if tt is not None and (best is None or tt < best):
                    best, kind = tt, s
            if plane_y is not None and d[1] < 0:
                tp = plane_y / d[1]
                if tp > 0 and (best is None or tp < best):
                    best, kind = tp, "plane"
            if best is None:
                row.append(background)
                continue
            p = (d[0] * best, d[1] * best, d[2] * best)
            if kind == "plane":
                nrm, albedo = (0.0, 1.0, 0.0), plane_albedo
            else:
                nrm = ((p[0] - kind[0]) / kind[3], (p[1] - kind[1]) / kind[3], (p[2] - kind[2]) / kind[3])
                albedo = kind[4]
            L = (lx - p[0], ly - p[1], lz - p[2])
            dist = math.sqrt(L[0] ** 2 + L[1] ** 2 + L[2] ** 2)
            L = (L[0] / dist, L[1] / dist, L[2] / dist)
            lam = max(0.0, nrm[0] * L[0] + nrm[1] * L[1] + nrm[2] * L[2])
            o = (p[0] + 1e-6 * nrm[0], p[1] + 1e-6 * nrm[1], p[2] + 1e-6 * nrm[2])
            lit = 1.0
            for s in spheres:
                tt = hit_sphere(o, L, s)
                if tt is not None and tt < dist:
                    lit = 0.0
                    break
            row.append(albedo * (ambient + (1 - ambient) * lit * lam))
        rows.append(row)
    return rows
