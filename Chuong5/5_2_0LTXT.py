import numpy as np

def power_method(A, y0, eps=1e-5, max_iter=100, precision=5):
    y_k = y0.astype(float)
    lambda_prev = 0.0
    for k in range(1, max_iter + 1):
        z = A @ y_k
        max_idx = np.argmax(np.abs(z))
        lambda_k = z[max_idx]
        
        # Cơ chế bảo vệ hội tụ: tránh chia cho 0 khi ma trận thặng dư tiến về ma trận không
        if abs(lambda_k) < 1e-12:
            return 0.0, y_k
            
        y_next = z / lambda_k
        error = abs(lambda_k - lambda_prev)
        if error < eps:
            return lambda_k, y_next
        y_k = y_next
        lambda_prev = lambda_k
    return lambda_prev, y_k

def deflation_general(A, lambda_k, v_k, y0_left, step, precision=5):
    """
    Phương pháp xuống thang tổng quát áp dụng cho ma trận KHÔNG đối xứng.
    Thuật toán bảo toàn nguyên trạng, tích hợp thêm tham số 'step' để định danh chu trình.
    """
    print("\n" + "=" * 85)
    print(f"{f'TIẾN HÀNH XUỐNG THANG TỔNG QUÁT (GENERAL DEFLATION) - CHU KỲ {step}':^85}")
    print("=" * 85)
    
    # 1. Tìm vector riêng trái (w_k) bằng Power Method trên ma trận chuyển vị A.T
    print(f"1. Trích xuất vector riêng trái w_{step} từ không gian chuyển vị A^T...")
    _, w_k = power_method(A.T, y0_left, precision=precision)
    print(f"   Vector riêng trái thô (w_{step}) = {np.round(w_k, precision)}")
    
    # 2. Chuẩn hóa vector riêng trái sao cho w_k^T * v_k = 1
    inner_product = np.dot(w_k, v_k)
    
    # Chống suy biến trong trường hợp không gian trực giao
    if abs(inner_product) < 1e-12:
        inner_product = 1e-12
        
    w_k_normalized = w_k / inner_product
    print(f"2. Cân bằng không gian (w_{step} / (w_{step}^T * v_{step})):")
    print(f"   Tích vô hướng w_{step}^T * v_{step} = {inner_product:.{precision}f}")
    print(f"   Vector riêng trái chuẩn hóa (w_{step}*) = {np.round(w_k_normalized, precision)}")
    
    # 3. Xây dựng ma trận chiếu
    projection_matrix = np.outer(v_k, w_k_normalized)
    
    # 4. Thiết lập ma trận thặng dư cho chu kỳ k+1
    A_next = A - lambda_k * projection_matrix
    
    print(f"\n3. Ma trận thặng dư (A_{step+1} = A_{step} - lambda_{step} * v_{step} * w_{step}^T):")
    print(np.round(A_next, precision))
    
    return A_next

# ==========================================
# KHỐI THỰC THI VÀ PHÂN TÍCH TRẠNG THÁI
# ==========================================
if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    
    A_asym = np.array([
        [2.0, 1.0, 3.0, 7.0],
        [0.0, 3.0, 4.0, 9.0],
        [1.0, 2.0, 5.0, 6.0],
        [2.0, 4.0, 7.0, 0.0]
    ])
    
    y_init = np.array([1.0, 1.0, 1.0, 1.0])
    
    print("MA TRẬN HỆ THỐNG GỐC (A_1):")
    print(A_asym)
    
    # Khởi tạo tham số không gian
    n_dimensions = A_asym.shape[0]
    A_current = A_asym.copy()
    
    # Cấu trúc lưu trữ dữ liệu phổ
    eigenvalues = []
    eigenvectors = []
    
    # Chu trình truy xuất đa trị riêng
    for k in range(1, n_dimensions + 1):
        print(f"\n>>> GIAI ĐOẠN {k}: TRÍCH XUẤT TRỊ RIÊNG THỨ {k}")
        lambda_k, v_k = power_method(A_current, y_init)
        
        print(f"[*] Phân tích hội tụ: lambda_{k} = {lambda_k:.5f}")
        print(f"[*] Vector riêng trạng thái v_{k} = {np.round(v_k, 5)}")
        
        eigenvalues.append(lambda_k)
        eigenvectors.append(v_k)
        
        # Áp dụng hàm xuống thang trừ khi đã đạt giới hạn chiều không gian
        if k < n_dimensions:
            # Điều kiện dừng sớm nếu trị riêng tiến về 0 (ma trận thặng dư đã hội tụ về ma trận không)
            if abs(lambda_k) < 1e-10:
                print(f"[*] Hệ thống ghi nhận ma trận thặng dư suy biến tại bước {k}. Ngắt chu trình xuống thang.")
                break
                
            A_current = deflation_general(A_current, lambda_k, v_k, y_init, step=k)
            
    # Kết xuất toàn bộ phổ dữ liệu
    print("\n" + "=" * 85)
    print(f"{'TỔNG HỢP TOÀN BỘ PHỔ TRỊ RIÊNG VÀ VECTOR RIÊNG':^85}")
    print("=" * 85)
    for i in range(len(eigenvalues)):
        print(f"Hệ trạng thái {i+1}: lambda = {eigenvalues[i]:>8.5f} | Vector = {np.round(eigenvectors[i], 5)}")