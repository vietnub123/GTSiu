import numpy as np

def print_extended_matrix(AI, n, title, val_fmt):
    """In ma trận mở rộng [A | I]"""
    print(f"\n--- {title} ---")
    for row in AI:
        left = "  ".join([f"{val:{val_fmt}}" for val in row[:n]])
        right = "  ".join([f"{val:{val_fmt}}" for val in row[n:]])
        print(f"[ {left}  |  {right} ]")

def find_inverse_gauss_jordan_verbose(A_input, val_fmt=".4f"):
    """
    Tìm ma trận nghịch đảo bằng Gauss-Jordan (Tối ưu bằng NumPy).
    val_fmt: Định dạng hiển thị số thực.
    """
    A = np.array(A_input, dtype=float)
    
    print("\n" + "="*80)
    print(" TÌM MA TRẬN NGHỊCH ĐẢO BẰNG GAUSS-JORDAN (NUMPY OPTIMIZED)")
    print("="*80)

    # --- BƯỚC BỔ SUNG: KIỂM TRA ĐIỀU KIỆN NGHỊCH ĐẢO ---
    print("\n>>> KIỂM TRA ĐIỀU KIỆN TỒN TẠI MA TRẬN NGHỊCH ĐẢO:")
    
    # 1. Kiểm tra ma trận vuông
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"[-] Lỗi: Ma trận đầu vào phải là ma trận vuông. Kích thước hiện tại: {A.shape}")
    print("[+] Kích thước hợp lệ: Ma trận vuông.")

    # 2. Kiểm tra định thức (Khác 0)
    det_A = np.linalg.det(A)
    print(f"[*] Định thức det(A) = {det_A:{val_fmt}}")
    if np.isclose(det_A, 0.0, atol=1e-12):
        raise ValueError("[-] Lỗi: Ma trận suy biến (Định thức = 0). Không tồn tại ma trận nghịch đảo!")
    print("[+] Định thức khác 0. Đảm bảo tồn tại ma trận nghịch đảo.")
    # ----------------------------------------------------

    n = A.shape[0]
    
    # 1. Tạo ma trận mở rộng [A | I] siêu tốc bằng np.hstack
    I = np.eye(n)
    AI = np.hstack((A, I))
    
    print_extended_matrix(AI, n, "Ma trận mở rộng [A | I] ban đầu", val_fmt)

    # 2. Bắt đầu khử Gauss-Jordan
    for k in range(n):
        print(f"\n>>> Xử lý cột {k}:")
        
        # Chọn phần tử chốt (Partial Pivoting)
        # np.argmax tìm vị trí phần tử lớn nhất trong cột k (từ dòng k trở đi)
        max_row = np.argmax(np.abs(AI[k:, k])) + k
        
        if max_row != k:
            # Hoán vị 2 dòng bằng Advanced Indexing của NumPy
            AI[[k, max_row]] = AI[[max_row, k]]
            print(f"[*] Đổi dòng {k} và dòng {max_row}")

        pivot = AI[k, k]
        # (Bước kiểm tra pivot này giữ lại để an toàn trước sai số làm tròn của máy tính)
        if abs(pivot) < 1e-12:
            raise ValueError("Ma trận suy biến trong quá trình biến đổi (Pivot=0).")

        # Chuẩn hóa dòng chốt để AI[k, k] = 1
        print(f"[*] Chuẩn hóa dòng {k}: Chia cho chốt = {pivot:{val_fmt}}")
        # Vector hóa: Chia toàn bộ các phần tử từ cột k trở đi cho pivot
        AI[k, k:] = AI[k, k:] / pivot
        
        # Khử tất cả các dòng khác (cả trên và dưới)
        for i in range(n):
            if i != k:
                factor = AI[i, k]
                if abs(factor) > 1e-12:
                    print(f"    -> Khử dòng {i}: D_{i} = D_{i} - ({factor:{val_fmt}}) * D_{k}")
                    # Vector hóa phép trừ 2 dòng (Thực hiện tính toán trên cả mảng)
                    AI[i, k:] = AI[i, k:] - factor * AI[k, k:]
        
        print_extended_matrix(AI, n, f"Ma trận sau bước {k}", val_fmt)

    # 3. Tách lấy ma trận nghịch đảo
    # Cắt lát (slice) lấy tất cả các dòng, và lấy các cột từ n trở đi
    A_inv = AI[:, n:]
    
    print("\n" + "="*80)
    print(" KẾT QUẢ MA TRẬN NGHỊCH ĐẢO A^-1:")
    for row in A_inv:
        print("[ " + "  ".join([f"{v:{val_fmt}}" for v in row]) + " ]")
    print("="*80)
    
    return A_inv

# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Trường hợp 1: Ma trận hợp lệ
    print("--- TEST 1: MA TRẬN HỢP LỆ ---")
    A_test_1 = [
        [2.0, 1.0, 1.0],
        [4.0, 3.0, 3.0],
        [8.0, 7.0, 9.0]
    ]
    try:
        find_inverse_gauss_jordan_verbose(A_test_1, val_fmt=".3f")
    except ValueError as e:
        print(e)

    # Trường hợp 2: Ma trận suy biến (Det = 0)
    print("\n--- TEST 2: MA TRẬN SUY BIẾN ---")
    A_test_2 = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0], # Dòng này tỷ lệ với dòng 1
        [7.0, 8.0, 9.0]
    ]
    try:
        find_inverse_gauss_jordan_verbose(A_test_2, val_fmt=".3f")
    except ValueError as e:
        print(e)