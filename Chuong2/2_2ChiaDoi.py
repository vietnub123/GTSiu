import math

def bisection_method(f, a, b, tol, max_iter=100):
    """
    Tìm nghiệm của phương trình f(x) = 0 bằng phương pháp chia đôi.
    
    Tham số:
    f        : Hàm mục tiêu (callable).
    a, b     : Các điểm mút của khoảng phân ly nghiệm ban đầu [a, b].
    tol      : Sai số cho phép (dung sai - tolerance).
    max_iter : Số bước lặp tối đa.
    
    Trả về:
    root     : Nghiệm gần đúng.
    n_iter   : Số bước lặp đã thực hiện.
    """
    # 1. Kiểm tra điều kiện đầu vào của định lý Bolzano-Cauchy
    if f(a) * f(b) >= 0:
        raise ValueError(f"Hàm f(x) phải đổi dấu trên đoạn [{a}, {b}]. f(a)={f(a)}, f(b)={f(b)}")

    # 2. Vòng lặp chia đôi
    for n_iter in range(1, max_iter + 1):
        # Tính trung điểm
        c = (a + b) / 2.0
        
        # Kiểm tra điều kiện dừng:
        # Cách 1: Nửa khoảng cách < sai số (đánh giá tiên nghiệm)
        # Cách 2: Giá trị tuyệt đối của f(c) xấp xỉ 0 (để tránh lỗi số học)
        if abs(b - a) / 2.0 < tol or abs(f(c)) < 1e-15:
            return c, n_iter
        
        # 3. Cập nhật khoảng nghiệm
        if f(a) * f(c) < 0:
            b = c  # Nghiệm nằm trong nửa trái [a, c]
        else:
            a = c  # Nghiệm nằm trong nửa phải [c, b]
            
    # Báo lỗi nếu vượt qua max_iter mà chưa đạt dung sai
    print("Cảnh báo: Đã đạt đến số bước lặp tối đa nhưng chưa đạt sai số mong muốn.")
    return (a + b) / 2.0, max_iter

# ==========================================
# VÍ DỤ SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    # Giải phương trình: x^3 - x - 1 = 0 trên đoạn [1, 2]
    def f_example(x):
        return x**3 - x - 1

    a_init, b_init = 1.0, 2.0
    tolerance = 1e-4

    try:
        nghiem, so_buoc_lap = bisection_method(f_example, a_init, b_init, tolerance)
        print(f"Nghiệm tìm được: x = {nghiem:.6f}")
        print(f"Số bước lặp: {so_buoc_lap}")
        print(f"Giá trị hàm tại nghiệm f(x) = {f_example(nghiem):.6e}")
    except ValueError as e:
        print(e)