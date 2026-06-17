import numpy as np

def print_matrix(M, title, val_fmt):
    """In ma trận theo định dạng định chuẩn khoa học"""
    print(f"\n--- {title} ---")
    for row in M:
        print("[ " + "  ".join([f"{v:{val_fmt}}" for v in row]) + " ]")

def danilevsky_method(A_input, val_fmt=".7f", verbose=True):
    """
    Phương pháp Danilevsky tìm đa thức đặc trưng với hệ thống giải thích chi tiết (Pedagogical Verbose).
    """
    A = np.array(A_input, dtype=float)
    n = A.shape[0]
    
    col_permutation = list(range(n))
    permutation_log = []
    
    if verbose:
        print("\n" + "═"*80)
        print(" KHỞI CHẠY THUẬT TOÁN DANILEVSKY TÌM ĐA THỨC ĐẶC TRƯNG")
        print(" Mục tiêu: Biến đổi đồng dạng ma trận A về dạng ma trận Frobenius (Đồng hành)")
        print(" Cơ sở toán học: A^(new) = M_inv * A * M")
        print("═"*80)
        print_matrix(A, "Ma trận khởi tạo A^(0)", val_fmt)

    for k in range(n - 2, -1, -1):
        target_row = k + 1
        step_num = n - 1 - k
        
        if verbose:
            print("\n" + "█"*80)
            print(f" BƯỚC {step_num} (k = {k}): THIẾT LẬP VECTOR CƠ SỞ TẠI DÒNG {target_row}")
            print("█"*80)
            print(f" [*] Mục tiêu vi mô: Biến đổi dòng {target_row} thành dạng [0, ..., 1, ..., 0] (với số 1 tại cột {k}).")
        
        # 1. KIỂM TRA PHẦN TỬ CHỐT VÀ HOÁN VỊ ĐỒNG DẠNG (Zero-Pivoting)
        if abs(A[target_row, k]) < 1e-15:
            if verbose:
                print(f" [!] CẢNH BÁO: Phần tử chốt a_{target_row},{k} = 0. Kích hoạt cơ chế tìm kiếm hoán vị.")
            
            pivot_col = k
            for j in range(k - 1, -1, -1):
                if abs(A[target_row, j]) > 1e-15:
                    pivot_col = j
                    break
            
            if pivot_col != k:
                if verbose:
                    print(f"  -> Tìm thấy phần tử thay thế hợp lệ tại cột {pivot_col}: a_{target_row},{pivot_col} = {A[target_row, pivot_col]:{val_fmt}}")
                    print(f"  -> Thực hiện phép biến đổi đồng dạng hoán vị: A' = P^-1 * A * P")
                    print(f"     (Hoán vị đồng thời cột {k} ↔ {pivot_col} và dòng {k} ↔ {pivot_col} để bảo toàn trị riêng)")
                
                A[:, [k, pivot_col]] = A[:, [pivot_col, k]]  # Hoán vị cột (A * P)
                A[[k, pivot_col], :] = A[[pivot_col, k], :]  # Hoán vị dòng (P^-1 * A * P)
                
                col_permutation[k], col_permutation[pivot_col] = col_permutation[pivot_col], col_permutation[k]
                permutation_log.append((k, pivot_col))
                
                if verbose:
                    print_matrix(A, f"Ma trận A sau khi hoán vị P^-1 * A * P", val_fmt)
            else:
                if verbose:
                    print(f" [LỖI TOÁN HỌC] Không tồn tại phần tử khả nghịch để tạo chốt cho dòng {target_row}.")
                    print(" Thuật toán Danilevsky cơ sở buộc phải dừng. Cần áp dụng phân rã khối (Block Decomposition).")
                return A, col_permutation, None, permutation_log
        
        pivot = A[target_row, k]
        if abs(pivot) < 1e-15:
            if verbose:
                print(f" [LỖI] Phần tử chốt suy biến: a_{target_row},{k} ≈ 0.")
            return A, col_permutation, None, permutation_log
        
        if verbose:
            print(f" [+] Xác nhận phần tử chốt (Pivot): a_{target_row},{k} = {pivot:{val_fmt}}")
        
        # 2. XÂY DỰNG MA TRẬN M_inv (Ma trận chứa các bản sao)
        if verbose:
            print(f"\n [*] Thiết lập ma trận M_inv (Ma trận nhân bên trái):")
            print(f"  -> Lý thuyết: Thay thế dòng {k} của ma trận đơn vị I bằng toàn bộ dòng {target_row} của ma trận A.")
        M_inv = np.eye(n)
        M_inv[k, :] = A[target_row, :]
        if verbose:
            print_matrix(M_inv, f"Ma trận M_inv_{step_num}", val_fmt)

        # 3. XÂY DỰNG MA TRẬN M (Ma trận chứa phép chia nghịch đảo)
        if verbose:
            print(f"\n [*] Thiết lập ma trận M (Ma trận nhân bên phải):")
            print(f"  -> Lý thuyết: Thay thế dòng {k} của ma trận đơn vị I.")
            print(f"  -> Phần tử tại cột chốt (k, k): 1 / {pivot:{val_fmt}}")
            print(f"  -> Các phần tử khác (k, j): -a_{target_row},j / {pivot:{val_fmt}}")
        M = np.eye(n)
        M[k, :] = -A[target_row, :] / pivot
        M[k, k] = 1.0 / pivot
        if verbose:
            print_matrix(M, f"Ma trận M_{step_num}", val_fmt)

        # 4. BIẾN ĐỔI ĐỒNG DẠNG
        if verbose:
            print(f"\n [*] Thực thi phép biến đổi đại số đồng dạng: A^({step_num}) = M_inv_{step_num} * A^({step_num-1}) * M_{step_num}")
        A = M_inv @ A @ M
        
        if verbose:
            if k == 0:
                print("\n [+] ĐÃ ĐẠT DẠNG CHUẨN FROBENIUS!")
                print_matrix(A, "MA TRẬN ĐỒNG HÀNH (FROBENIUS MATRIX F)", val_fmt)
            else:
                print_matrix(A, f"Ma trận kết quả A^({step_num})", val_fmt)

    # 5. TRÍCH XUẤT ĐA THỨC ĐẶC TRƯNG
    if verbose:
        print("\n" + "═"*80)
        print(" TỔNG HỢP KẾT QUẢ ĐA THỨC ĐẶC TRƯNG")
        print("═"*80)
    
    p_coeffs = A[0].copy()
    
    if verbose:
        print(f" [*] Trích xuất hệ số từ dòng 1 của ma trận Frobenius (p_1, p_2, ..., p_n):")
        print(f"  -> {np.array([round(p, 7) for p in p_coeffs])}")
        
        terms = [f"λ^{n}"]
        for i, p in enumerate(p_coeffs):
            power = n - 1 - i
            sign = "-" if p > 0 else "+"
            term = f"{sign} {abs(p):{val_fmt}}"
            
            if power > 0:
                term += f"*λ^{power}" if power > 1 else "*λ"
            terms.append(term)
        
        poly_str = " ".join(terms)
        
        # Xác định dấu của định thức đa thức tùy thuộc vào cấp ma trận (lý thuyết P(λ) = det(A - λI) hoặc det(λI - A))
        det_sign = "-" if n % 2 != 0 else ""
        
        print(f"\n [*] Phương trình đặc trưng P(λ) = det(A - λI):")
        print(f"  => P(λ) = {det_sign}({poly_str})")
        
        if permutation_log:
            print(f"\n [*] Lưu ý về không gian vector:")
            print(f"  -> Các phép hoán vị đã thực hiện: {permutation_log}")
            print(f"  -> Thứ tự không gian cột hiện tại: {col_permutation}")
    
    return A, col_permutation, p_coeffs, permutation_log

# ==========================================
# GIAO THỨC KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True)

    print("\n" + "█"*80)
    print("█ TEST CƠ SỞ ĐẠI SỐ (Hệ ma trận 3x3 đối chiếu tính tay)")
    print("█"*80)
    A3 = [
        [1, 2, 3],
        [2, 1, 4],
        [3, 1, 1]
    ]
    F3, perm3, poly3, log3 = danilevsky_method(A3)
    
    # Kiểm chứng chéo với np.linalg.eigvals để đảm bảo tính đúng đắn phổ
    eigenvalues_original = np.linalg.eigvals(A3)
    roots_from_poly = np.roots(np.insert(-poly3, 0, 1)) if poly3 is not None else []
    
    print("\n" + "═"*80)
    print(" BƯỚC ĐỐI CHIẾU NGHIỆM PHỔ TRỊ RIÊNG (KIỂM CHỨNG TOÁN HỌC)")
    print("═"*80)
    print(f"  - Tập trị riêng gốc từ hàm thư viện (np.linalg.eigvals): \n    {np.sort(eigenvalues_original)}")
    print(f"  - Tập trị riêng từ nghiệm đa thức Danilevsky (np.roots): \n    {np.sort(roots_from_poly)}")