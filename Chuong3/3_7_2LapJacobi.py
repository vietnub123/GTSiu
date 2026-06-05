import numpy as np

def check_diagonally_dominant(A):
    """
    Kiểm tra tính chéo trội nghiêm ngặt của ma trận bằng NumPy.
    """
    n = A.shape[0]
    # Lấy giá trị tuyệt đối của đường chéo chính
    diag_vals = np.abs(np.diag(A))
    
    # Tính tổng giá trị tuyệt đối của từng hàng
    sum_rows = np.sum(np.abs(A), axis=1) - diag_vals
    
    # Nếu có phần tử đường chéo bằng 0
    if np.any(diag_vals == 0):
        return False, float('inf')
        
    # Tính mảng q_i = tổng hàng (trừ chéo) / giá trị chéo
    q_array = sum_rows / diag_vals
    q_max = np.max(q_array)
    
    # Kiểm tra xem TẤT CẢ các tổng hàng có nhỏ hơn giá trị chéo không
    is_dominant = np.all(sum_rows < diag_vals)
    
    return is_dominant, q_max

def print_formatted_matrix(M, title, val_fmt):
    print(f"\n--- {title} ---")
    for row in M:
        print("[ " + "  ".join([f"{v:{val_fmt}}" for v in row]) + " ]")

def jacobi_inverse_verbose(A_input, tol=1e-5, max_iter=50, val_fmt=".4f", err_fmt=".6f"):
    """
    Tìm ma trận nghịch đảo bằng phương pháp lặp Jacobi (Sử dụng NumPy).
    Giải phương trình ma trận AX = I.
    """
    A = np.array(A_input, dtype=float)
    n = A.shape[0]
    
    print("\n" + "="*80)
    print(" BẮT ĐẦU TÌM MA TRẬN NGHỊCH ĐẢO BẰNG LẶP JACOBI (AX = I)")
    print("="*80)

    # 1. Kiểm tra điều kiện chéo trội
    is_dominant, q = check_diagonally_dominant(A)
    if is_dominant:
        print(f"[+] Ma trận chéo trội nghiêm ngặt (q = {q:{val_fmt}}). Phương pháp chắc chắn hội tụ.")
    else:
        print(f"[-] CẢNH BÁO: Ma trận không chéo trội (q = {q:{val_fmt}}). Có thể phân kỳ!")

    # 2. Chuẩn bị các ma trận D, R và I
    # Lấy đường chéo chính của A
    diag_A = np.diag(A)
    if np.any(np.abs(diag_A) < 1e-12):
        raise ValueError("Phần tử trên đường chéo bằng 0, không thể dùng Jacobi.")
        
    # Tạo ma trận D^(-1) bằng cách nghịch đảo các phần tử trên đường chéo
    D_inv = np.diag(1.0 / diag_A)
    
    # Ma trận R = A - D (Lấy A trừ đi đường chéo của nó)
    R = A.copy()
    np.fill_diagonal(R, 0)
    
    # Ma trận đơn vị I
    I = np.eye(n)

    # 3. Khởi tạo X_curr (Chọn X_0 = D^-1 để bớt được 1 vòng lặp)
    X_curr = D_inv.copy()
    print_formatted_matrix(X_curr, "Ma trận khởi tạo X^(0) = D^(-1)", val_fmt)

    # 4. Quá trình lặp: X^(k+1) = D^(-1) * (I - R * X^(k))
    for k in range(1, max_iter + 1):
        print(f"\n>>> BƯỚC LẶP k = {k}")
        
        # Dùng toán tử @ để nhân ma trận, - để trừ ma trận
        RX = R @ X_curr
        I_minus_RX = I - RX
        X_next = D_inv @ I_minus_RX

        # Tính sai số bằng chuẩn vô cùng (chuẩn hàng)
        diff = X_next - X_curr
        error = np.linalg.norm(diff, np.inf)

        # Cập nhật
        X_curr = X_next.copy()

        print_formatted_matrix(X_curr, f"Ma trận X^({k})", val_fmt)
        print(f"=> Sai số ||X^({k}) - X^({k-1})|| = {error:{err_fmt}}")

        if error < tol:
            print("\n" + "="*80)
            print(f" THUẬT TOÁN HỘI TỤ SAU {k} BƯỚC LẶP (Sai số < {tol:{err_fmt}})")
            print_formatted_matrix(X_curr, "MA TRẬN NGHỊCH ĐẢO GẦN ĐÚNG (A^-1)", val_fmt)
            print("="*80)
            return X_curr

    print("\nCẢNH BÁO: Đạt giới hạn vòng lặp tối đa nhưng chưa hội tụ.")
    return X_curr

# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    A_mat = [
        [12.0, 2.0, -1],
        [0, 10, 2],
        [1, 1, -6]
    ]
    
    try:
        jacobi_inverse_verbose(
            A_mat, 
            tol=1e-5, 
            max_iter=20, 
            val_fmt=".5f", 
            err_fmt=".2e"
        )
    except ValueError as e:
        print(e)