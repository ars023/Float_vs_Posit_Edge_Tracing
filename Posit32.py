import numpy as np
import softposit as sp
from softposit import quire32, posit32
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plane(qx, qy, qz, sx, sy, sz, tx, ty, tz):
    # Convert each coordinate individually to posit32
    qx, qy, qz = sp.posit32(qx), sp.posit32(qy), sp.posit32(qz)
    sx, sy, sz = sp.posit32(sx), sp.posit32(sy), sp.posit32(sz)
    tx, ty, tz = sp.posit32(tx), sp.posit32(ty), sp.posit32(tz)

    # Initialize quires for z1, z2, z3, z4
    q_z1 = quire32()
    q_z2 = quire32()
    q_z3 = quire32()
    q_z4 = quire32()

    # Compute x and y components
    x1, x2, x3 = sx - qx, sy - qy, sz - qz
    y1, y2, y3 = tx - qx, ty - qy, tz - qz

    # Compute z1 = (x2 * y3) - (x3 * y2)
    q_z1.qma(x2, y3)
    q_z1.qma(-x3, y2)
    z1 = q_z1.toPosit()

    # Compute z2 = (x3 * y1) - (x1 * y3)
    q_z2.qma(x3, y1)
    q_z2.qma(-x1, y3)
    z2 = q_z2.toPosit()

    # Compute z3 = (x1 * y2) - (x2 * y1)
    q_z3.qma(x1, y2)
    q_z3.qma(-x2, y1)
    z3 = q_z3.toPosit()

    # Compute z4 = (qx * z1) + (qy * z2) + (qz * z3)
    q_z4.qma(qx, z1)
    q_z4.qma(qy, z2)
    q_z4.qma(qz, z3)
    z4 = q_z4.toPosit()

    return z1, z2, z3, z4


def intersect(px, py, pz, rx, ry, rz, z1, z2, z3, z4):
    # Helper function to safely initialize posit32
    def safe_posit32(value):
        if not isinstance(value, sp.posit32):
            return sp.posit32(value)
        return value

    # Convert inputs to safe posit32
    px, py, pz = safe_posit32(px), safe_posit32(py), safe_posit32(pz)
    rx, ry, rz = safe_posit32(rx), safe_posit32(ry), safe_posit32(rz)
    z1, z2, z3, z4 = safe_posit32(z1), safe_posit32(z2), safe_posit32(z3), safe_posit32(z4)

    # Initialize quires for h = z4 - ((px * z1) + (py * z2) + (pz * z3))
    q_h = quire32()
    q_h.qma(px, z1)  # Add px * z1
    q_h.qma(py, z2)  # Add py * z2
    q_h.qma(pz, z3)  # Add pz * z3
    h = z4 - q_h.toPosit()  # Subtract from z4 and convert to posit32

    # Initialize quires for i denominator = ((rx * z1) + (ry * z2) + (rz * z3))
    q_denom = quire32()
    q_denom.qma(rx, z1)  # Add rx * z1
    q_denom.qma(ry, z2)  # Add ry * z2
    q_denom.qma(rz, z3)  # Add rz * z3
    denom = q_denom.toPosit()  # Convert to posit32

    # Compute i = h / denom
    i = h / denom

    # Compute ix, iy, iz using quires for better accuracy
    q_ix = quire32()
    q_ix.qma(i, rx)  # Add i * rx
    q_ix.qma(px, sp.posit32(1))  # Add px
    ix = q_ix.toPosit()  # Convert to posit32

    q_iy = quire32()
    q_iy.qma(i, ry)  # Add i * ry
    q_iy.qma(py, sp.posit32(1))  # Add py
    iy = q_iy.toPosit()  # Convert to posit32

    q_iz = quire32()
    q_iz.qma(i, rz)  # Add i * rz
    q_iz.qma(pz, sp.posit32(1))  # Add pz
    iz = q_iz.toPosit()  # Convert to posit32

    return ix, iy, iz

def cross_product_quire(a, b):
    
    # Initialize quires for each component of the result
    q_cx, q_cy, q_cz = quire32(), quire32(), quire32()

    # Compute c_x = a_y * b_z - a_z * b_y
    q_cx.qma(a[1], b[2])  # Add a_y * b_z to the quire
    q_cx.qma(a[2], posit32(-1) * b[1])  # Subtract a_z * b_y from the quire

    # Compute c_y = a_z * b_x - a_x * b_z
    q_cy.qma(a[2], b[0])  # Add a_z * b_x to the quire
    q_cy.qma(a[0], posit32(-1) * b[2])  # Subtract a_x * b_z from the quire

    # Compute c_z = a_x * b_y - a_y * b_x
    q_cz.qma(a[0], b[1])  # Add a_x * b_y to the quire
    q_cz.qma(a[1], posit32(-1) * b[0])  # Subtract a_y * b_x from the quire

    # Convert quires back to posit32 for the result
    c_x, c_y, c_z = q_cx.toPosit(), q_cy.toPosit(), q_cz.toPosit()

    return [c_x, c_y, c_z]


def exact_dot_product_posit(v1, v2):
    q = sp.quire32()
    for x, y in zip(v1, v2):
        q.qma(x, y)
    result = q.toPosit()
    return result

def is_point_in_triangle(p, a, b, c):
    ab = [(b[i] - a[i]) for i in range(3)]
    bc = [(c[i] - b[i]) for i in range(3)]
    ca = [(a[i] - c[i]) for i in range(3)]

    ap = [(p[i] - a[i]) for i in range(3)]
    bp = [(p[i] - b[i]) for i in range(3)]
    cp = [(p[i] - c[i]) for i in range(3)]
    
    cross1 = cross_product_quire(ab, ap)
    cross2 = cross_product_quire(bc, bp)
    cross3 = cross_product_quire(ca, cp)

    if exact_dot_product_posit(cross1, cross2) >= 0 and exact_dot_product_posit(cross2, cross3) >= 0:
        return True
    else:
        return False

# Triangle vertices

qx, qy, qz = 0.5, 0.5, 0.5
sx, sy, sz = 1, 0.25, 1
tx, ty, tz = -0.125, -0.5, -0.25
z1, z2, z3, z4 = plane(qx, qy, qz, sx, sy, sz, tx, ty, tz)

# Endpoints for ray shooting
p_start = np.array([1, 1, 1])
p_end = np.array([0.5, -0.25, 0.5])
n_steps = 8  # Number of steps between p_start and p_end

# Initialize counters for intersections
intersect_count = 0
no_intersect_count = 0
skipped_count = 0
outside_box = 0
intersection_points = []

# Loop over steps to create rays
for t in np.linspace(0, 1, n_steps):
    # Calculate the point along the line segment
    p_current = p_start + t * p_end
    px, py, pz = p_current[0], p_current[1], p_current[2]
    rx, ry, rz = -0.5, -0.5, -0.5

    # Calculate the denominator
    denominator = (rx * z1) + (ry * z2) + (rz * z3)
    if denominator == 0:
        if z4 - ((px * z1) + (py * z2) + (pz * z3)) == 0:
            skipped_count += 1
            continue  # Ray is in the plane, skip it
    else:
        # Calculate intersection point
        ix, iy, iz = intersect(px, py, pz, rx, ry, rz, z1, z2, z3, z4)
        
        # Check if intersection point is within bounding box
        if (min(qx, sx, tx) <= ix <= max(qx, sx, tx) and 
            min(qy, sy, ty) <= iy <= max(qy, sy, ty) and 
            min(qz, sz, tz) <= iz <= max(qz, sz, tz)):
            
            # Check if the point is within the triangle
            if is_point_in_triangle([ix, iy, iz], [qx, qy, qz], [sx, sy, sz], [tx, ty, tz]):
                intersect_count += 1
                print(f"{[ix, iy, iz]} yes")
                intersection_points.append([ix, iy, iz])
            else:
                #print(f"{p_current }")
                no_intersect_count += 1
                print(f"{[ix, iy, iz]} no")
        else:
            outside_box += 1

# Print the final counts
print(f"Rays that intersect the triangle: {intersect_count}")
print(f"Rays that do not intersect the triangle: {no_intersect_count}")
print(f"Rays that were skipped (in the plane): {skipped_count}")
print(f"Rays outside the bounding box: {outside_box}")
print(f"Total rays checked: {intersect_count + no_intersect_count + skipped_count}")

# Convert intersection points to numpy array and remove duplicates
if intersection_points:
    intersection_points = np.unique(np.array(intersection_points, dtype=np.float32), axis=0)

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the triangle
triangle = np.array([[qx, qy, qz], [sx, sy, sz], [tx, ty, tz], [qx, qy, qz]], dtype=np.float32)
ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 'b-', label='Triangle')

# Plot rays
for t in np.linspace(0, 1, n_steps, dtype=np.float32):
    ray_start_t = p_start + t * (p_end - p_start)
    t_range = np.linspace(-2, 2, 100, dtype=np.float32)
    ray_points = ray_start_t + np.outer(t_range, [rx, ry, rz])
    ax.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], 'r-', alpha=0.3)

# Plot intersection points
if len(intersection_points) > 0:
    ax.scatter(intersection_points[:, 0], intersection_points[:, 1], intersection_points[:, 2], 
               color='g', s=50, alpha=1.0, label='Intersection')

# Set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Ray and Triangle Intersections with Posit32')

# Rotation and axes
ax.view_init(elev=30, azim=45)

ax.set_xlim([0.4, 1])
ax.set_ylim([0, 0.6])
ax.set_zlim([0.4, 1])

plt.legend()
ax.view_init(elev=40, azim=45)  # Additional rotation
plt.show()
