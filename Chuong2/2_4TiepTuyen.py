from typing import Callable, Tuple, Optional

def newton_raphson_academic(
    f: Callable[[float], float], 
    df: Callable[[float], float], 
    x0: float, 
    tol: float = 1e-6, 
    max_iter: int = 50, 
    val_fmt: str = ".6f"
) -> Tuple[Optional[float], int, float]:
    """
    Phương pháp Newton-Raphson dành cho học tập và phân tích giải tích số.
    
    Tham số:
        f (Callable): Hàm số f(x).
        df (Callable): Đạo hàm f'(x).
        x0 (float): Giá trị khởi tạo.
        tol (float): Sai số cho phép (Tiêu chí dừng: |x_{n+1} - x_n| < tol).
        max_iter (int): Số vòng lặp tối đa.
        val_fmt (str): Định dạng in số (VD: '.6f' cho 6 số thập phân, '.4e' cho ký pháp khoa học).
    """
    print("\n" + "="*85)
    print(" PHƯƠNG PHÁP TIẾP TUYẾN (NEWTON-RAPHSON)")
    print("="*85)
    print(f"Tham số cài đặt: x0 = {x0}, Sai số (tol) = {tol}, Max Iter = {max_iter}")
    print("-" * 85)
    
    # Tiêu đề bảng theo dõi các bước lặp
    header = f"{'Bước (n)':<10} | {'x_n':<15} | {'f(x_n)':<15} | {'f''(x_n)':<15} | {'Sai số (Δx)':<15}"
    print(header)
    print("-" * 85)

    x_n = x0

    for n in range(1, max_iter + 1):
        fx_n = f(x_n)
        dfx_n = df(x_n)

        # Kiểm tra đạo hàm triệt tiêu để tránh chia cho 0
        if abs(dfx_n) < 1e-14:
            print("-" * 85)
            print(f"[-] DỪNG KHẨN CẤP: Đạo hàm f'(x) ≈ 0 tại bước {n}. Không thể tiếp tục.")
            return None, n, 0.0

        # Tính toán x_{n+1}
        x_next = x_n - (fx_n / dfx_n)
        
        # Tính sai số tuyệt đối
        error = abs(x_next - x_n)

        # In thông số từng bước lặp với định dạng tùy chỉnh (val_fmt)
        row_str = (
            f"{n:<10} | "
            f"{x_n:<15{val_fmt}} | "
            f"{fx_n:<15{val_fmt}} | "
            f"{dfx_n:<15{val_fmt}} | "
            f"{error:<15{val_fmt}}"
        )
        print(row_str)

        # Kiểm tra tiêu chí hội tụ
        if error < tol:
            print("-" * 85)
            print(f"[+] HỘI TỤ THÀNH CÔNG tại bước {n}.")
            print(f"[+] Nghiệm gần đúng: x* ≈ {x_next:{val_fmt}}")
            print(f"[+] Sai số cuối cùng: {error:{val_fmt}} < {tol}")
            return x_next, n, error

        # Cập nhật giá trị cho bước lặp tiếp theo
        x_n = x_next

    # Nếu chạy hết vòng lặp mà chưa đạt sai số mong muốn
    print("-" * 85)
    print(f"[-] THẤT BẠI: Thuật toán không phân kỳ/hội tụ đủ nhanh sau {max_iter} bước.")
    return None, max_iter, error

# ==========================================
# CÁCH SỬ DỤNG VÀ THAY ĐỔI ĐỊNH DẠNG
# ==========================================
if __name__ == "__main__":
    import math

    # Ví dụ: Giải phương trình f(x) = x - cos(x) = 0
    def f(x):
        return x - math.cos(x)

    def df(x):
        return 1 + math.sin(x)

    # 1. In với 6 chữ số thập phân (Bình thường)
    print("\n>>> KIỂM THỬ 1: Định dạng 6 chữ số thập phân (val_fmt='.6f')")
    newton_raphson_academic(f, df, x0=1.0, tol=1e-5, val_fmt=".6f")

