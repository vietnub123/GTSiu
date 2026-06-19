import numpy as np

def solve_least_squares_svd(A, b, epsilon=1e-12, precision=4, verbose=True):
    """
    Giải bài toán bình phương tối thiểu ||Ax - b||_2 -> min bằng thuật toán SVD.
    
    Tham số:
        A (list/ndarray): Ma trận hệ số kích thước m x n.
        b (list/ndarray): Vector hằng số kích thước m x 1.
        epsilon (float): Ngưỡng dung sai loại bỏ các giá trị kỳ dị xấp xỉ không.
        precision (int): Số chữ số thập phân hiển thị ở output.
        verbose (bool): Kích hoạt chế độ in chi tiết từng bước tính toán.
        
    Trả về:
        x (ndarray): Vector nghiệm cực tiểu hóa chuẩn thặng dư.
    """
    # Thiết lập định dạng hiển thị cho hệ thống numpy
    np.set_printoptions(precision=precision, suppress=True)

    # Khởi tạo và chuẩn hóa cấu trúc dữ liệu mảng
    A_mat = np.array(A, dtype=float)
    b_vec = np.array(b, dtype=float).reshape(-1, 1)
    m, n = A_mat.shape

    if verbose:
        print("=" * 60)
        print("KHỞI TẠO THÔNG SỐ ĐẦU VÀO")
        print("-" * 60)
        print(f"Ma trận hệ số A ({m}x{n}):\n{A_mat}")
        print(f"Vector hằng số b ({m}x1):\n{b_vec}")
        print(f"Ngưỡng dung sai số học (epsilon): {epsilon}\n")

    # BƯỚC 1: Thực hiện phân tích SVD
    # Hàm np.linalg.svd trả về U, các phần tử đường chéo của Sigma, và V^T (chuyển vị của V)
    U, s, Vh = np.linalg.svd(A_mat, full_matrices=True)
    
    # Tái tạo cấu trúc ma trận Sigma (m x n) phục vụ mục đích tường minh hóa số liệu
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, s)

    if verbose:
        print("=" * 60)
        print("BƯỚC 1: PHÂN TÍCH GIÁ TRỊ KỲ DỊ (A = U * Sigma * V^T)")
        print("-" * 60)
        print(f"Ma trận trực giao U ({m}x{m}):\n{U}\n")
        print(f"Ma trận đường chéo Sigma ({m}x{n}):\n{Sigma}\n")
        print(f"Ma trận trực giao V^T ({n}x{n}):\n{Vh}\n")

    # BƯỚC 2: Tính toán vector hình chiếu không gian c = U^T * b
    c = np.dot(U.T, b_vec)

    if verbose:
        print("=" * 60)
        print("BƯỚC 2: TÍNH VECTOR HÌNH CHIẾU (c = U^T * b)")
        print("-" * 60)
        print(f"Vector c:\n{c}\n")

    # BƯỚC 3 & 4: Giải hệ tọa độ biến đổi y = Sigma^+ * c với cơ chế điều chuẩn
    y = np.zeros((n, 1))
    rank_estimate = min(m, n)
    
    for i in range(rank_estimate):
        # Kiểm tra ngưỡng dung sai để đảm bảo tính ổn định số học
        if s[i] > epsilon:
            y[i, 0] = c[i, 0] / s[i]
        else:
            y[i, 0] = 0.0  # Triệt tiêu nhiễu từ các trị kỳ dị suy biến

    if verbose:
        print("=" * 60)
        print("BƯỚC 3 & 4: XÁC ĐỊNH NGHIỆM KHÔNG GIAN BIẾN ĐỔI (y = Sigma^+ * c)")
        print("-" * 60)
        print(f"Phổ giá trị kỳ dị (Sigma_ii): {s}")
        print(f"Vector y:\n{y}\n")

    # BƯỚC 5: Truy hồi nghiệm hệ tọa độ gốc x = V * y
    V = Vh.T  # Chuyển vị lại Vh để thu được V
    x = np.dot(V, y)

    if verbose:
        print("=" * 60)
        print("BƯỚC 5: TRUY HỒI NGHIỆM HỆ TỌA ĐỘ GỐC (x = V * y)")
        print("-" * 60)
        print(f"Vector nghiệm x:\n{x}\n")
        
        # Đánh giá sai số hậu nghiệm
        residual_vec = np.dot(A_mat, x) - b_vec
        residual_norm = np.linalg.norm(residual_vec)
        print("=" * 60)
        print("ĐÁNH GIÁ ĐỘ LỆCH CHUẨN (THẶNG DƯ)")
        print("-" * 60)
        print(f"Vector thặng dư (Ax - b):\n{residual_vec}")
        print(f"Chuẩn Euclid thặng dư ||Ax - b||_2: {residual_norm:.{precision}f}")
        print("=" * 60)
    return x

# =====================================================================
# THỰC THI KHỐI MÃ VÍ DỤ 
# Đối chiếu với bài toán số trị đã chứng minh giải tích
# =====================================================================
if __name__ == "__main__":
    A_example = [
        [1,  1,5],
        [1, -1,7],
        [0,  0,9]
    ]
    
    b_example = [
        [2],
        [0],
        [1]
    ]
    
    # Khởi chạy thuật toán với sai số 1e-10, hiển thị 4 chữ số thập phân
    x_optimal = solve_least_squares_svd(
        A=A_example, 
        b=b_example, 
        epsilon=1e-10, 
        precision=4, 
        verbose=True
    )