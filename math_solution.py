import numpy as np


def triangulation(
        camera_matrix: np.ndarray,
        camera_position1: np.ndarray,
        camera_rotation1: np.ndarray,
        camera_position2: np.ndarray,
        camera_rotation2: np.ndarray,
        image_points1: np.ndarray,
        image_points2: np.ndarray
):
    """
    :param camera_matrix: first and second camera matrix, np.ndarray 3x3
    :param camera_position1: first camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation1: first camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param camera_position2: second camera position in world coordinate system, np.ndarray 3x1
    :param camera_rotation2: second camera rotation matrix in world coordinate system, np.ndarray 3x3
    :param image_points1: points in the first image, np.ndarray Nx2
    :param image_points2: points in the second image, np.ndarray Nx2
    :return: triangulated points, np.ndarray Nx3
    """

    # Ensure camera positions are column vectors of shape (3,1)
    C1 = camera_position1.reshape(3, 1)
    C2 = camera_position2.reshape(3, 1)

    # Transpose the rotation matrices
    R1 = camera_rotation1.T
    R2 = camera_rotation2.T

    # Compute translation vectors with transposed rotations
    t1 = -R1 @ C1
    t2 = -R2 @ C2

    # Compute projection matrices with transposed rotations
    P1 = camera_matrix @ np.hstack((R1, t1))
    P2 = camera_matrix @ np.hstack((R2, t2))

    # Number of points
    N = image_points1.shape[0]

    # Initialize array for 3D points
    points_3d = np.zeros((N, 3))

    for i in range(N):
        u1, v1 = image_points1[i]
        u2, v2 = image_points2[i]

        # Construct matrix A for the ith point
        A = np.array([
            (u1 * P1[2, :] - P1[0, :]),
            (v1 * P1[2, :] - P1[1, :]),
            (u2 * P2[2, :] - P2[0, :]),
            (v2 * P2[2, :] - P2[1, :])
        ])

        # Solve A * X = 0 using SVD
        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        X_homogeneous /= X_homogeneous[3]  # Normalize homogeneous coordinate

        # Store the 3D point
        points_3d[i] = X_homogeneous[:3]

    return points_3d
