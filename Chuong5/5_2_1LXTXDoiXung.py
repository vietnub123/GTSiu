import numpy as np

def power_method_symmetric(A, y0, eps=1e-6, max_iter=100, precision=5):
    """
    Thuật toán Phương pháp Lũy thừa (Power Method) ưu tiên tính minh bạch.
    Hiển thị đầy đủ sai số tại mỗi bước lặp để theo dõi tốc độ hội tụ.
    """
    y_k = y0.astype(float)
    lambda_prev = 0.0
    
    print(f"\n--- BẮT ĐẦU CHU TRÌNH LẶP TÌM TRỊ RIÊNG ---")
    print(f"    Mục tiêu sai số (eps): {eps}")
    
    for k in range(1, max_iter + 1):
        # Nhân ma trận với vector trạng thái hiện tại
        z = A @ y_k
        
        # Xác định phần tử có giá trị tuyệt đối lớn nhất làm trị riêng xấp xỉ
        max_idx = np.argmax(np.abs(z))
        lambda_k = z[max_idx]
        
        # Bảo vệ tính phân kỳ nếu không gian tiến về không
        if abs(lambda_k) < 1e-12:
            print(f"    [Bước {k:02d}] Cảnh báo: Trị riêng xấp xỉ 0. Dừng chu trình.")
            return 0.0, y_k
            
        # Chuẩn hóa vector cho bước lặp tiếp theo
        y_next = z / lambda_k
        
        # Đánh giá sai số tuyệt đối
        error_current = abs(lambda_k - lambda_prev)
        
        # Xuất dữ liệu lặp trung gian
        print(f"    [Bước {k:02d}] Trị riêng xấp xỉ: {lambda_k:.{precision}f} | Sai số: {error_current:.10f}")
        
        # Kiểm tra điều kiện hội tụ
        if error_current < eps:
            print(f"    => HỆ THỐNG HỘI TỤ TẠI BƯỚC {k} (Sai số < {eps})")
            
            # Chuẩn hóa Euclid (L2 norm) cho vector riêng để phục vụ trực giao hóa
            l2_norm = np.linalg.norm(y_next)
            y_final_normalized = y_next / l2_norm if l2_norm > 1e-12 else y_next
            
            return lambda_k, y_final_normalized
            
        # Cập nhật trạng thái cho vòng lặp mới
        y_k = y_next
        lambda_prev = lambda_k
        
    print("    => CẢNH BÁO: Thuật toán không hội tụ trong số bước lặp cho phép.")
    return lambda_prev, y_k

def deflation_symmetric(A, lambda_k, v_k, step, precision=5):
    """
    Phương pháp xuống thang đối xứng (Hotelling's Deflation).
    Trực tiếp dùng tích tensor của vector riêng phải (do v_left = v_right).
    """
    print("\n" + "=" * 80)
    print(f"{f'TIẾN HÀNH XUỐNG THANG ĐỐI XỨNG - CHU KỲ {step}':^80}")
    print("=" * 80)
    
    # 1. Tính toán chuẩn bình phương của vector riêng
    inner_product = np.dot(v_k, v_k)
    print(f"1. Tích vô hướng vector (v_{step}^T * v_{step}) = {inner_product:.{precision}f}")
    
    # 2. Xây dựng ma trận chiếu (Tích ngoài v_k * v_k^T)
    projection_matrix = np.outer(v_k, v_k) / inner_product
    print(f"\n2. Ma trận toán tử chiếu [ (v_{step} * v_{step}^T) / chuẩn ]:")
    print(np.round(projection_matrix, precision))
    
    # 3. Tính ma trận thặng dư trực giao
    A_next = A - lambda_k * projection_matrix
    print(f"\n3. Ma trận thặng dư (A_{step+1} = A_{step} - lambda_{step} * toán_tử_chiếu):")
    print(np.round(A_next, precision))
    
    return A_next

# ==========================================
# KHỐI THỰC THI VÀ KIỂM CHỨNG TOÁN HỌC
# ==========================================
if __name__ == "__main__":
    # Thiết lập tham số hiển thị mảng numpy
    np.set_printoptions(precision=5, suppress=True)
    
    # Khai báo ma trận đối xứng thực A = A^T
    A_sym = np.array([
        [4.0, 1.0, -1.0],
        [1.0, 3.0, 2.0],
        [-1.0, 2.0, 5.0]
    ])
    
    # Vector trạng thái khởi tạo
    y_initial = np.array([1.0, 1.0, 1.0])
    
    print("MA TRẬN HỆ THỐNG ĐỐI XỨNG GỐC (A_1):")
    print(A_sym)
    
    # Tham số không gian cấu trúc
    n_dim = A_sym.shape[0]
    A_current = A_sym.copy()
    
    # Thiết lập biến lưu trữ quang phổ hệ thống
    spectral_values = []
    eigen_vectors = []
    
    # Giới hạn dung sai lỗi
    tolerance_eps = 1e-6
    
    # Khởi động chu trình phân rã phổ không gian
    for k in range(1, n_dim + 1):
        print(f"\n>>> GIAI ĐOẠN {k}: TÌM KIẾM TRỊ RIÊNG THỨ {k}")
        
        # Khởi chạy thuật toán lũy thừa
        lambda_k, v_k = power_method_symmetric(A_current, y_initial, eps=tolerance_eps)
        
        print(f"[*] Kết quả trích xuất hệ thống {k}:")
        print(f"    - Trị riêng lambda_{k} = {lambda_k:.6f}")
        print(f"    - Vector riêng chuẩn hóa v_{k} = {np.round(v_k, 5)}")
        
        spectral_values.append(lambda_k)
        eigen_vectors.append(v_k)
        
        # Triệt tiêu không gian đã giải quyết nếu chưa đạt chiều cuối
        if k < n_dim:
            if abs(lambda_k) < 1e-10:
                print(f"[*] Ma trận thặng dư tiến về 0 tại bậc {k}. Kết thúc chu trình trực giao hóa.")
                break
                
            A_current = deflation_symmetric(A_current, lambda_k, v_k, step=k)

    # Tổng hợp báo cáo phân tích phổ ma trận
    print("\n" + "=" * 80)
    print(f"{'KẾT QUẢ ĐẦY ĐỦ CỦA PHÂN RÃ QUANG PHỔ':^80}")
    print("=" * 80)
    for i in range(len(spectral_values)):
        print(f"Hệ trạng thái {i+1} | Trị riêng: {spectral_values[i]:>8.5f} | Vector không gian: {np.round(eigen_vectors[i], 5)}")