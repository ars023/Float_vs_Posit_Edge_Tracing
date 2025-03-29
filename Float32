import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plane(qx, qy, qz, sx, sy, sz, tx, ty, tz):
    qx, qy, qz = np.float32(qx), np.float32(qy), np.float32(qz)
    sx, sy, sz = np.float32(sx), np.float32(sy), np.float32(sz)
    tx, ty, tz = np.float32(tx), np.float32(ty), np.float32(tz)

    x1 = sx - qx
    x2 = sy - qy
    x3 = sz - qz
    y1 = tx - qx
    y2 = ty - qy
    y3 = tz - qz
    z1 = (x2 * y3) - (x3 * y2)
    z2 = (x3 * y1) - (x1 * y3)
    z3 = (x1 * y2) - (x2 * y1)
    z4 = (qx * z1) + (qy * z2) + (qz * z3)
    return z1, z2, z3, z4

def intersect(px, py, pz, rx, ry, rz, z1, z2, z3, z4):
    px, py, pz = np.float32(px), np.float32(py), np.float32(pz)
    rx, ry, rz = np.float32(rx), np.float32(ry), np.float32(rz)
    z1, z2, z3, z4 = np.float32([z1, z2, z3, z4])

    h = z4 - ((px * z1) + (py * z2) + (pz * z3))
    i = h / ((rx * z1) + (ry * z2) + (rz * z3))
    ix = i * rx + px
    iy = i * ry + py
    iz = i * rz + pz
    return ix, iy, iz

def vector_cross_product(v1, v2):
    return np.cross(np.float32(v1), np.float32(v2))

def is_point_in_triangle(p, a, b, c):
    ab = np.array(b, dtype=np.float32) - np.array(a, dtype=np.float32)
    bc = np.array(c, dtype=np.float32) - np.array(b, dtype=np.float32)
    ca = np.array(a, dtype=np.float32) - np.array(c, dtype=np.float32)
    
    ap = np.array(p, dtype=np.float32) - np.array(a, dtype=np.float32)
    bp = np.array(p, dtype=np.float32) - np.array(b, dtype=np.float32)
    cp = np.array(p, dtype=np.float32) - np.array(c, dtype=np.float32)
    
    cross1 = vector_cross_product(ab, ap)
    cross2 = vector_cross_product(bc, bp)
    cross3 = vector_cross_product(ca, cp)
    
    if np.dot(cross1, cross2) >= 0 and np.dot(cross2, cross3) >= 0:
        return True
    else:
        return False

# Triangle vertices
qx, qy, qz = np.float32([0.5, 0.5, 0.5])
sx, sy, sz = np.float32([1, 0.25, 1])
tx, ty, tz = np.float32([-0.125, -0.5, -0.25])
z1, z2, z3, z4 = plane(qx, qy, qz, sx, sy, sz, tx, ty, tz)

# Endpoints for ray shooting
p_start = np.array([1, 1, 1], dtype=np.float32)
p_end = np.array([0.5, -0.25, 0.5], dtype=np.float32)
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
    px, py, pz = p_current
    rx, ry, rz = -0.5, -0.5, -0.5

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
                no_intersect_count += 1
                print(f"{[ix, iy, iz]} no")
        else:
            outside_box += 1

# Print the final counts
print(f"Rays that intersect the triangle: {intersect_count}")
print(f"Rays that do not intersect the triangle: {no_intersect_count}")
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
               color='g', s=50, alpha = 1.0, label='Intersection')

# Set labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Ray and Triangle Intersections with Float32')

# Rotation and axes
ax.view_init(elev=30, azim=45)

ax.set_xlim([0.4, 1])
ax.set_ylim([0, 0.6])
ax.set_zlim([0.4, 1])

plt.legend()
ax.view_init(elev=40, azim=45)  # Additional rotation as needed
plt.show()
