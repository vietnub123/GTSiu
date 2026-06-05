import numpy as np

def svd_power_iteration_verbose(A, tol=1e-5, max_iter=50, val_fmt=".7f", err_fmt=".7e"):
    """
    Tìm giá trị kỳ dị trội (sigma) và cặp vector kỳ dị trái/phải (u, v) 
    của ma trận A bằng phương pháp lũy thừa luân phiên, có in chi tiết các bước lặp.
    """
    m, n = A.shape
    # Khởi tạo vector v toàn số 1, chuẩn hóa về chuẩn 2
    v_curr = np.ones(n)
    v_curr = v_curr / np.linalg.norm(v_curr)   #np.linalg.norm(diff, np.inf)
    
    sigma_curr = 0.0
    
    print(f"  [*] Vector khởi tạo v^(0) = [{', '.join([f'{val:{val_fmt}}' for val in v_curr])}]")

    for k in range(1, max_iter + 1):
        # 1. Tìm vector u xấp xỉ
        u_next = A @ v_curr
        u_next = u_next / np.linalg.norm(u_next)
        
        # 2. Tìm vector v xấp xỉ
        v_next = A.T @ u_next
        v_next = v_next / np.linalg.norm(v_next)
        
        # 3. Tính giá trị kỳ dị sigma xấp xỉ
        sigma_next = np.linalg.norm(A @ v_next)
        # 4. Tính sai số (dựa trên vector v hoặc sigma)
        error = np.linalg.norm(v_curr - v_next)
        
        # In chi tiết từng bước lặp
        print(f"\n  >>> BƯỚC LẶP k = {k}")
        print(f"      Vector u^({k}) = [{', '.join([f'{val:{val_fmt}}' for val in u_next])}]")
        print(f"      Trị kỳ dị sigma^({k}) = {sigma_next:{val_fmt}}")
        print(f"      Vector v^({k}) = [{', '.join([f'{val:{val_fmt}}' for val in v_next])}]")
        print(f"      => Sai số ||v^({k}) - v^({k-1})|| = {error:{err_fmt}}")
        
        # Kiểm tra dừng
        if error < tol:
            print(f"\n  => HỘI TỤ SAU {k} BƯỚC LẶP")
            u_final = (A @ v_next) / sigma_next
            return u_final, sigma_next, v_next
            
        v_curr = v_next
        sigma_curr = sigma_next

    print("\n  [-] Đạt số bước lặp tối đa nhưng chưa hội tụ!")
    u_final = (A @ v_curr) / sigma_curr
    return u_final, sigma_curr, v_curr

def svd_deflation_verbose(A, tol=1e-7, val_fmt=".7f"):
    """
    Tìm toàn bộ phân rã SVD (U, Sigma, V^T) bằng phương pháp xuống thang.
    """
    m, n = A.shape
    k_components = min(m, n)
    
    A_curr = A.astype(float).copy()
    
    U_cols = []
    Sigmas = []
    V_rows = []
    
    print("\n" + "="*80)
    print(f" BẮT ĐẦU PHÂN TÍCH GIÁ TRỊ KỲ DỊ (SVD) BẰNG PHƯƠNG PHÁP LŨY THỪA")
    print("="*80)
    
    for i in range(k_components):
        print(f"\n" + "-"*60)
        print(f" TÌM THÀNH PHẦN KỲ DỊ THỨ {i+1}")
        print("-"*60)
        
        # 1. Gọi hàm lũy thừa có in chi tiết
        u, sigma, v = svd_power_iteration_verbose(A_curr, tol=tol, val_fmt=val_fmt)
        
        # Điều kiện dừng nếu ma trận đã bị triệt tiêu hết
        if sigma < 1e-10:
            print(f"\n[*] Thành phần kỳ dị quá nhỏ ({sigma:.2e} ~ 0). Đã triệt tiêu hết ma trận!")
            break
            
        U_cols.append(u)
        Sigmas.append(sigma)
        V_rows.append(v)
        
        # 2. Xuống thang
        rank_1_matrix = sigma * np.outer(u, v)
        A_curr = A_curr - rank_1_matrix
        
        print(f"\n[*] Ma trận A sau khi xuống thang bước {i+1} (Chuẩn phần dư = {np.linalg.norm(A_curr):{val_fmt}}):")
        for row in A_curr:
            print("    [" + "  ".join([f"{val:{val_fmt}}" for val in row]) + "]")

    # Xây dựng các ma trận
    U = np.column_stack(U_cols)
    S = np.diag(Sigmas)
    
    # [ĐÃ SỬA LỖI DEPRECATION] Thay row_stack bằng vstack
    Vt = np.vstack(V_rows)
    
    return U, S, Vt

# ==========================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True)

    # Sử dụng một ma trận 3x2 để test
    A_rect = np.array([
        [3,3,7],
        [3,9,10],
        [8,4,6],
        [2,10,9]
    ])
    
    print("MA TRẬN GỐC A:")
    print(A_rect)

    # Chạy thuật toán
    U, Sigma, Vt = svd_deflation_verbose(A_rect, tol=1e-6)
    
    print("\n" + "="*80)
    print(" KẾT QUẢ ĐỊNH LÝ PHÂN RÃ (A = U * Sigma * V^T)")
    print("="*80)
    print("1. Ma trận U:\n", U)
    print("\n2. Ma trận Sigma:\n", Sigma)
    print("\n3. Ma trận V^T:\n", Vt)
    
    A_reconstruct = U @ Sigma @ Vt
    print("\nMa trận A phục hồi:\n", A_reconstruct)
    print(f"=> Sai số: {np.linalg.norm(A_rect - A_reconstruct, 1):.7e}")

    #np.linalg.norm(diff, np.inf) chuẩn hàng hoặc chuẩn max nếu là vector đơn
    #np.linalg.norm(diff, 1) chuẩn cột hoặc chuẩn tổng nếu là vector đơn
    #np.linalg.norm(A, 'fro') chuẩn Frobenius
    #np.linalg.norm(A, 2) chuẩn spectral (giá trị kỳ dị lớn nhất) $$||A||_2 = \sigma_{max}(A) = \sqrt{\lambda_{max}(A^T A)}$$