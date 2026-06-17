import numpy as np

# (Hàm power_method giữ nguyên từ đoạn code trước)
def power_method(A, y0, eps=1e-5, max_iter=100, precision=5):
    y_k = y0.astype(float)
    lambda_prev = 0.0
    for k in range(1, max_iter + 1):
        z = A @ y_k
        max_idx = np.argmax(np.abs(z))
        lambda_k = z[max_idx]
        y_next = z / lambda_k
        error = abs(lambda_k - lambda_prev)
        if error < eps:
            return lambda_k, y_next
        y_k = y_next
        lambda_prev = lambda_k
    return lambda_prev, y_k

def deflation_general(A, lambda1, v1, y0_left, precision=5):
    """
    Phương pháp xuống thang tổng quát áp dụng cho ma trận KHÔNG đối xứng.
    Yêu cầu tính toán thêm vector riêng trái.
    """
    print("\n" + "=" * 85)
    print(f"{'TIẾN HÀNH XUỐNG THANG TỔNG QUÁT (GENERAL DEFLATION)':^85}")
    print("=" * 85)
    
    # 1. Tìm vector riêng trái (w1) bằng Power Method trên ma trận chuyển vị A.T
    print("1. Trích xuất vector riêng trái từ không gian chuyển vị A^T...")
    _, w1 = power_method(A.T, y0_left, precision=precision)
    print(f"   Vector riêng trái thô (w1) = {np.round(w1, precision)}")
    
    # 2. Chuẩn hóa vector riêng trái sao cho w1^T * v1 = 1
    inner_product = np.dot(w1, v1)
    w1_normalized = w1 / inner_product
    print(f"2. Cân bằng không gian (w1 / (w1^T * v1)):")
    print(f"   Tích vô hướng w1^T * v1 = {inner_product:.{precision}f}")
    print(f"   Vector riêng trái chuẩn hóa (w1*) = {np.round(w1_normalized, precision)}")
    
    # 3. Xây dựng ma trận chiếu: v1 * w1^T
    projection_matrix = np.outer(v1, w1_normalized)
    
    # 4. Thiết lập không gian suy biến B
    B = A - lambda1 * projection_matrix
    
    print("\n3. Ma trận không gian suy biến (B = A - lambda1 * v1 * w1^T):")
    print(np.round(B, precision))
    
    return B

# ==========================================
# KIỂM CHỨNG VỚI MA TRẬN KHÔNG ĐỐI XỨNG
# ==========================================
if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    
    # Ma trận KHÔNG đối xứng
    # Phổ trị riêng giải tích: lambda = [3, 1]
    A_asym = np.array([
        [2.0, 1.0],
        [0.0, 3.0]
    ])
    
    y_init = np.array([1.0, 1.0])
    
    print("MA TRẬN HỆ THỐNG KHÔNG ĐỐI XỨNG (A):")
    print(A_asym)
    
    # GIAI ĐOẠN 1: Tìm trị riêng trội thứ nhất
    print("\n>>> GIAI ĐOẠN 1: TÌM TRỊ RIÊNG TRỘI")
    lambda_1, v_1 = power_method(A_asym, y_init)
    print(f"[*] Kết quả: lambda_1 = {lambda_1:.5f}, Vector riêng phải v_1 = {np.round(v_1, 5)}")
    
    # GIAI ĐOẠN 2: Xuống thang tổng quát
    B_matrix = deflation_general(A_asym, lambda_1, v_1, y_init)
    
    # GIAI ĐOẠN 3: Tìm trị riêng thứ hai từ ma trận B
    print("\n>>> GIAI ĐOẠN 3: TÌM TRỊ RIÊNG THỨ HAI")
    lambda_2, v_2 = power_method(B_matrix, y_init)
    print(f"[*] Kết quả: lambda_2 = {lambda_2:.5f}, Vector riêng v_2 = {np.round(v_2, 5)}")