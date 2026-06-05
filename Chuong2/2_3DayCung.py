def secant_method(f, x0, x1, tol=1e-5, max_iter=100):
    """
    Giải phương trình f(x) = 0 bằng phương pháp dây cung (Secant Method).
    
    Tham số:
    - f: Hàm số cần tìm nghiệm (dạng hàm Python trả về giá trị số).
    - x0, x1: Hai giá trị dự đoán ban đầu.
    - tol: Sai số chấp nhận được (điều kiện dừng).
    - max_iter: Số vòng lặp tối đa để tránh lặp vô hạn.
    
    Trả về:
    - Nghiệm gần đúng của phương trình, hoặc None nếu không hội tụ.
    """
    print(f"{'Vòng lặp':<10} | {'x_n':<15} | {'f(x_n)':<15}")
    print("-" * 45)
    
    for i in range(max_iter):
        f_x0 = f(x0)
        f_x1 = f(x1)
        
        # In ra các bước để theo dõi
        print(f"{i:<10} | {x1:<15.6f} | {f_x1:<15.6f}")
        
        # Tránh chia cho 0 nếu f(x1) và f(x0) quá gần nhau
        if abs(f_x1 - f_x0) < 1e-14:
            print("Lỗi: Mẫu số quá nhỏ, không thể tiếp tục tính toán.")
            return None
        
        # Công thức phương pháp dây cung
        x_next = x1 - f_x1 * ((x1 - x0) / (f_x1 - f_x0))
        
        # Kiểm tra điều kiện hội tụ (sai số tuyệt đối giữa 2 bước lặp hoặc f(x) tiến về 0)
        if abs(x_next - x1) < tol or abs(f(x_next)) < tol:
            print("-" * 45)
            print(f"Hội tụ tại nghiệm: {x_next:.6f} sau {i+1} vòng lặp.")
            return x_next
        
        # Cập nhật giá trị cho vòng lặp tiếp theo
        x0 = x1
        x1 = x_next
        
    print("Cảnh báo: Thuật toán không đạt được độ chính xác mong muốn sau số vòng lặp tối đa.")
    return x1

# ================================
# VÍ DỤ SỬ DỤNG
# Giải phương trình: f(x) = x^3 - x - 1 = 0
# ================================
if __name__ == "__main__":
    # Định nghĩa hàm f(x)
    def f_example(x):
        return x**3 - x - 1
    
    # Hai giá trị dự đoán ban đầu
    x_init_0 = 1.0
    x_init_1 = 2.0
    
    # Gọi hàm
    nghiem = secant_method(f_example, x_init_0, x_init_1, tol=1e-5)