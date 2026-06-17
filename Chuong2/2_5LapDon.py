import math

def fixed_point_iteration(g, x0, q=None, eps=1e-4, max_iter=100, error_criterion='posteriori', precision=5):
    """
    Thực thi thuật toán lặp đơn tìm điểm bất động của hàm g(x).
    
    Tham số:
    - g: Hàm lặp g(x) (đối tượng callable).
    - x0: Điểm khởi tạo x_0.
    - q: Hệ số co (0 < q < 1). Bắt buộc nếu dùng tiêu chuẩn sai số hậu nghiệm.
    - eps: Dung sai cho phép (epsilon).
    - max_iter: Số vòng lặp tối đa.
    - error_criterion: Chuẩn sai số ('posteriori' - Hậu nghiệm, 'simple' - Đơn giản).
    - precision: Độ chính xác khi hiển thị kết quả (số chữ số thập phân).
    
    Trả về:
    - Nghiệm xấp xỉ x* (float) hoặc None nếu phân kỳ/không hội tụ.
    """
    
    # Thiết lập chuỗi định dạng hiển thị dựa trên tham số precision
    fmt = f".{precision}f"
    fmt_err = f".{precision+2}f" # Hiển thị sai số chi tiết hơn
    
    print("=" * 85)
    print(f"{'BẮT ĐẦU QUÁ TRÌNH LẶP ĐƠN (FIXED-POINT ITERATION)':^85}")
    print("=" * 85)
    print(f"Giá trị khởi tạo x_0 = {x0}")
    print(f"Dung sai epsilon    = {eps}")
    print(f"Tiêu chuẩn sai số   = {error_criterion}")
    
    # Xác định ngưỡng hội tụ và hệ số nhân sai số tùy thuộc vào tiêu chuẩn
    if error_criterion == 'posteriori':
        if q is None or q <= 0 or q >= 1:
            raise ValueError("Tiêu chuẩn hậu nghiệm yêu cầu hệ số co q trong khoảng (0, 1).")
        error_multiplier = q / (1 - q)
        print(f"Hệ số co q          = {q}")
        print(f"Ngưỡng hội tụ chuẩn = eps * (1-q)/q = {eps / error_multiplier:{fmt_err}}")
    elif error_criterion == 'simple':
        error_multiplier = 1.0
        print("Ngưỡng hội tụ chuẩn = epsilon")
    else:
        raise ValueError("error_criterion chỉ nhận 'posteriori' hoặc 'simple'.")

    print("-" * 85)
    print(f"{'Bước (k)':<10} | {'x_k':<15} | {'x_{k+1} = g(x_k)':<20} | {'Sai số Delta':<20}")
    print("-" * 85)

    x_k = x0
    
    for k in range(max_iter):
        try:
            # Tính toán giá trị xấp xỉ tiếp theo
            x_next = g(x_k)
        except Exception as e:
            print(f"\n[-] Lỗi tính toán tại chu kỳ {k}: {e}")
            return None
        
        # Đo lường khoảng cách giữa hai xấp xỉ liên tiếp
        diff = abs(x_next - x_k)
        
        # Đánh giá sai số dựa trên chuẩn đã chọn
        current_error = error_multiplier * diff
        
        # Kết xuất thông tin chu kỳ hiện tại
        print(f"{k:<10} | {x_k:<15{fmt}} | {x_next:<20{fmt}} | {current_error:<20{fmt_err}}")
        
        # Kiểm tra tiêu chuẩn hội tụ
        if current_error <= eps:
            print("-" * 85)
            print(f"[*] Hệ thống đạt trạng thái hội tụ sau {k+1} bước lặp.")
            print(f"[*] Nghiệm xấp xỉ x^* = {x_next:{fmt}}")
            print(f"[*] Sai số cuối cùng  = {current_error:{fmt_err}} <= {eps}")
            print("=" * 85)
            return x_next
        
        # Cập nhật trạng thái cho chu kỳ tiếp theo
        x_k = x_next

    # Xử lý trường hợp vượt giới hạn vòng lặp mà không hội tụ
    print("-" * 85)
    print(f"[-] Cảnh báo: Trạng thái phân kỳ hoặc không hội tụ sau {max_iter} bước lặp.")
    print("=" * 85)
    return None

# ==========================================
# GIAO THỨC CHẠY CHƯƠNG TRÌNH & KIỂM CHỨNG
# ==========================================
if __name__ == "__main__":
    # Bài toán: f(x) = x^3 - x - 1 = 0 trên đoạn [1, 2].
    # Thiết lập hàm lặp tương đương x = (x + 1)^(1/3)
    def g_func(x):
        return (x + 1) ** (1/3)
    
    # Cấu hình các tham số vật lý và toán học của hệ thống
    x_init = 1.5
    epsilon = 1e-4
    
    # Tính toán cực đại đạo hàm trên [1, 2] theo lý thuyết: q = g'(1) ≈ 0.210
    q_coeff = 0.210  
    
    # Kịch bản 1: Sử dụng chuẩn sai số hậu nghiệm (khuyến nghị cho độ chính xác cao)
    print("\n>>> KỊCH BẢN 1: ÁP DỤNG CHUẨN SAI SỐ HẬU NGHIỆM")
    root_1 = fixed_point_iteration(
        g=g_func, 
        x0=x_init, 
        q=q_coeff, 
        eps=epsilon, 
        error_criterion='posteriori',
        precision=5
    )
    
    # Kịch bản 2: Sử dụng chuẩn sai số đơn giản |x_{k+1} - x_k| <= eps
    print("\n>>> KỊCH BẢN 2: ÁP DỤNG CHUẨN SAI SỐ ĐƠN GIẢN (HIỆU SỐ TRỰC TIẾP)")
    root_2 = fixed_point_iteration(
        g=g_func, 
        x0=x_init, 
        eps=epsilon, 
        error_criterion='simple',
        precision=5
    )