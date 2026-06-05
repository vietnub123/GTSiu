import numpy as np

def check_convergence_conditions(A):
    """
    Kiểm tra các điều kiện hội tụ của Gauss-Seidel (Tối ưu bằng NumPy).
    1. Chéo trội nghiêm ngặt.
    2. Đối xứng và Xác định dương.
    """
    # 1. Kiểm tra chéo trội (Vectorization 100%)
    diag_vals = np.abs(np.diag(A))
    sum_rows = np.sum(np.abs(A), axis=1) - diag_vals
    is_dominant = np.all(sum_rows < diag_vals)
            
    # 2. Kiểm tra đối xứng bằng np.allclose
    is_symmetric = np.allclose(A, A.T, atol=1e-9)
    
    # 3. Kiểm tra xác định dương (Nếu đối xứng thì kiểm tra trị riêng)
    is_positive_definite = False
    if is_symmetric:
        eigenvalues = np.linalg.eigvals(A)
        if np.all(eigenvalues > 0):
            is_positive_definite = True
                
    return is_dominant, is_symmetric, is_positive_definite

def gauss_seidel_solve_verbose(A_input, B_input, x0=None, tol=1e-4, max_iter=50, val_fmt=".4f", err_fmt=".6f"):
    """
    Giải hệ phương trình đại số tuyến tính Ax = B bằng phương pháp lặp Gauss-Seidel.
    """
    # --- 1. CHUYỂN ĐỔI VÀ KIỂM TRA ĐIỀU KIỆN ĐẦU VÀO (VALIDATION) ---
    A = np.array(A_input, dtype=float)
    B = np.array(B_input, dtype=float)
    
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Lỗi: Ma trận A phải là ma trận vuông 2 chiều. Kích thước hiện tại: {A.shape}")
        
    n = A.shape[0]
    
    if B.ndim != 1 or B.shape[0] != n:
        raise ValueError(f"Lỗi: Kích thước vector B ({B.shape}) không khớp với số hàng của ma trận A ({n}).")
        
    # Kiểm tra phần tử trên đường chéo chính có bằng 0 hay không
    diag_elements = np.diag(A)
    if np.any(np.isclose(diag_elements, 0.0, atol=1e-12)):
        zero_indices = np.where(np.isclose(diag_elements, 0.0, atol=1e-12))[0]
        print(f"\n[!] CẢNH BÁO NGUY HIỂM: Đường chéo chính có phần tử bằng 0 tại các dòng: {zero_indices}")
        print("Thuật toán Gauss-Seidel không thể thực hiện trực tiếp (Lỗi chia cho 0). Hãy đổi chỗ các hàng!")
        return None

    x_curr = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    if x_curr.shape[0] != n:
        raise ValueError(f"Lỗi: Vector khởi tạo x0 có kích thước {x_curr.shape[0]}, yêu cầu phải bằng {n}.")
        
    # --- 2. IN THÔNG TIN BAN ĐẦU VÀ KIỂM TRA HỘI TỤ ---
    print("\n" + "="*80)
    print(" BẮT ĐẦU PHƯƠNG PHÁP LẶP GAUSS-SEIDEL")
    print("="*80)
    
    print("\n>>> HỆ PHƯƠNG TRÌNH ĐẦU VÀO [A | B]:")
    for i in range(n):
        row_str = "  ".join([f"{A[i, j]:{val_fmt}}" for j in range(n)])
        print(f"    [ {row_str} ] * [ x_{i} ] = [ {B[i]:{val_fmt}} ]")
    
    print("\n>>> ĐÁNH GIÁ ĐIỀU KIỆN HỘI TỤ:")
    is_dominant, is_symmetric, is_pd = check_convergence_conditions(A)
    if is_dominant:
        print("[+] Ma trận chéo trội nghiêm ngặt. Thuật toán CHẮC CHẮN HỘI TỤ.")
    elif is_symmetric and is_pd:
        print("[+] Ma trận đối xứng và xác định dương. Thuật toán CHẮC CHẮN HỘI TỤ.")
    elif is_symmetric and not is_pd:
        print("[-] CẢNH BÁO: Ma trận đối xứng nhưng KHÔNG xác định dương. Có thể không hội tụ!")
    else:
        print("[-] CẢNH BÁO: Ma trận không chéo trội, không đối xứng xác định dương. Có thể phân kỳ!")

    # --- TÍNH TOÁN VÀ IN HỆ SỐ CO q ---
    print("\n>>> PHÂN TÍCH MA TRẬN LẶP & HỆ SỐ CO q:")
    D_L = np.tril(A)      # Ma trận tam giác dưới bao gồm cả đường chéo (D + L)
    U = np.triu(A, 1)     # Ma trận tam giác trên (chỉ phần U)
    
    try:
        D_L_inv = np.linalg.inv(D_L)
        T_g = -np.dot(D_L_inv, U)
        q = np.linalg.norm(T_g, np.inf)
        print(f"[*] Hệ số co q (chuẩn vô cùng của ma trận lặp T_g) = {q:.6f}")
        if q < 1:
            print(f"    -> ĐÁNH GIÁ: Vì q = {q:.6f} < 1, thuật toán CHẮC CHẮN HỘI TỤ.")
        else:
            print(f"    -> ĐÁNH GIÁ: q = {q:.6f} >= 1, chuẩn vô cùng không đảm bảo hội tụ (nhưng vẫn có thể hội tụ nếu bán kính phổ < 1).")
    except np.linalg.LinAlgError:
        print("[!] Không thể tính nghịch đảo ma trận (D+L) để tìm hệ số co q.")

    print("\n>>> CÔNG THỨC LẶP TOÁN HỌC TỔNG QUÁT TỪNG ẨN:")
    for i in range(n):
        eq_parts = [f"({-A[i, j]:{val_fmt}} * x_{j})" for j in range(n) if j != i]
        print(f"    x_{i}^(k+1) = (1/{A[i, i]:{val_fmt}}) * [ {B[i]:{val_fmt}} + {' + '.join(eq_parts)} ]")

    print("\n" + "-"*80)
    print(f"[*] Vector xấp xỉ ban đầu x^(0) = [{', '.join([f'{v:{val_fmt}}' for v in x_curr])}]")
    print("-" * 80)

    # --- 3. VÒNG LẶP GAUSS-SEIDEL ---
    for k in range(1, max_iter + 1):
        print(f"\n>>> BƯỚC LẶP k = {k} <<<")
        x_prev = x_curr.copy() 
        
        for i in range(n):
            calc_str_parts = []
            for j in range(n):
                if i != j:
                    status = "(mới)" if j < i else "(cũ)"
                    calc_str_parts.append(f"({-A[i, j]:{val_fmt}} * {x_curr[j]:{val_fmt}}{status})")
            
            # Tính toán cập nhật
            sum_Ax = np.dot(A[i, :], x_curr) - A[i, i] * x_curr[i]
            x_curr[i] = (B[i] - sum_Ax) / A[i, i]
            
            calc_str = " + ".join(calc_str_parts)
            print(f"[*] x_{i}^({k}) = (1/{A[i, i]:{val_fmt}}) * [ {B[i]:{val_fmt}} + {calc_str} ] = {x_curr[i]:{val_fmt}}")

        # Đánh giá sai số
        error = np.linalg.norm(x_curr - x_prev, np.inf)
        
        print(f" => Vector kết quả bước k={k}: x^({k}) = [{', '.join([f'{v:{val_fmt}}' for v in x_curr])}]")
        print(f" => Sai số ||x^({k}) - x^({k-1})||_inf = {error:{err_fmt}}")

        # Kiểm tra điều kiện dừng
        if error < tol:
            print("\n" + "="*80)
            print(f" THUẬT TOÁN ĐÃ HỘI TỤ THÀNH CÔNG SAU {k} BƯỚC LẶP")
            print(f" (Sai số đạt được: {error:{err_fmt}} < Tiêu chuẩn dừng: {tol:{err_fmt}})")
            print(f" KẾT LUẬN NGHIỆM X = [{', '.join([f'{v:{val_fmt}}' for v in x_curr])}]")
            print("="*80)
            return x_curr

    print(f"\n[!] CẢNH BÁO: Đạt tới số bước lặp tối đa ({max_iter}) nhưng chưa thỏa mãn sai số!")
    print(f"Giá trị nghiệm tạm thời tại bước {max_iter}: [{', '.join([f'{v:{val_fmt}}' for v in x_curr])}]")
    return x_curr

# ==========================================
# KHỐI CHẠY KIỂM THỬ
# ==========================================
if __name__ == "__main__":
    # Ví dụ: Ma trận chéo trội nghiêm ngặt
    A_gs = [
        [ 0.04-1, 0.07, -0.01, 0,    -0.05],
        [-0.01, 0.04-1, -0.02, -0.01, 0.04],
        [-0.05, -0.04, 0.06-1, 0.03,  0.03],
        [-0.1 , 0.09 , 0.06, 0.04-1, -0.07],
        [-0.08, -0.1 ,-0.07, 0.05, -0.08-1]
    ]
    B_gs = [-7,6,-4,1,-7]
    
    gauss_seidel_solve_verbose(
        A_gs, 
        B_gs, 
        x0=[0.0, 0.0, 0.0, 0, 0], 
        tol=1e-5, 
        max_iter=15, 
        val_fmt=".7f",   
        err_fmt=".7e"    
    )