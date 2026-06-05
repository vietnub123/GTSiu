import numpy as np

def calculate_X0_pan_schreiber(A):
    """
    Tính ma trận khởi tạo X0 theo công thức Pan-Schreiber:
    X0 = A^T / (||A||_1 * ||A||_inf)
    """
    # Sử dụng np.linalg.norm để tính chuẩn 1 và chuẩn vô cùng
    norm_1 = np.linalg.norm(A, 1)
    norm_inf = np.linalg.norm(A, np.inf)
    
    denominator = norm_1 * norm_inf
    
    if denominator == 0:
        raise ValueError("Ma trận A có chuẩn bằng 0 (ma trận không), không thể tìm nghịch đảo.")
        
    # Toán tử / trên mảng NumPy tự động chia từng phần tử
    return A.T / denominator

def print_formatted_matrix(M, title, val_fmt):
    print(f"\n--- {title} ---")
    for row in M:
        print("[ " + "  ".join([f"{v:{val_fmt}}" for v in row]) + " ]")

def inverse_iteration_verbose(A, X0, tol=1e-6, max_iter=10, val_fmt=".5f", err_fmt=".3e"):
    """
    Tìm gần đúng ma trận nghịch đảo bằng phương pháp lặp Newton (Schulz).
    A: Ma trận cần tìm nghịch đảo (NumPy array).
    X0: Ma trận dự đoán ban đầu (NumPy array).
    """
    n = A.shape[0]
    X_curr = X0.copy()
    
    # Tạo ma trận đơn vị I bằng np.eye
    I = np.eye(n)

    print("\n" + "="*85)
    print(" BẮT ĐẦU PHƯƠNG PHÁP LẶP NEWTON TÌM MA TRẬN NGHỊCH ĐẢO")
    print("="*85)

    # Kiểm tra điều kiện hội tụ ban đầu (Dùng toán tử @ để nhân ma trận)
    AX0 = A @ X_curr
    E0 = I - AX0
    norm_E0 = np.linalg.norm(E0, np.inf)
    
    print(f"[*] Kiểm tra điều kiện hội tụ ban đầu: ||I - AX0|| = {norm_E0:{val_fmt}}")
    if norm_E0 >= 1:
        print("[-] CẢNH BÁO: Chuẩn sai số >= 1, phương pháp có thể không hội tụ!")
    else:
        print("[+] Chuẩn sai số < 1, phương pháp chắc chắn hội tụ bậc hai.")

    for k in range(1, max_iter + 1):
        print(f"\n>>> BƯỚC LẶP k = {k}")
        
        # Công thức: X_next = X_curr * (2I - A * X_curr)
        AX = A @ X_curr
        R = 2.0 * I - AX
        X_next = X_curr @ R
        
        # Tính sai số để kiểm tra điều kiện dừng
        diff = X_next - X_curr
        error = np.linalg.norm(diff, np.inf)
        
        X_curr = X_next.copy()
        
        print_formatted_matrix(X_curr, f"Ma trận X^({k})", val_fmt)
        print(f"=> Độ thay đổi ||X^({k}) - X^({k-1})|| = {error:{err_fmt}}")
        
        if error < tol:
            print("\n" + "="*85)
            print(f" THUẬT TOÁN HỘI TỤ SAU {k} BƯỚC LẶP")
            print_formatted_matrix(X_curr, "MA TRẬN NGHỊCH ĐẢO GẦN ĐÚNG CUỐI CÙNG", val_fmt)
            print("="*85)
            return X_curr

    print("\nCẢNH BÁO: Đạt giới hạn vòng lặp tối đa.")
    return X_curr

# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Khai báo ma trận bằng np.array
    A_mat = np.array([
        [ 1,6,5,10,-3],
        [-9,1,-10,5,5],
        [1,3,16,8,2],
        [-10,6,-10,1,2],
        [17,-8,4,3,-6]
    ])
    
    # Tính X0 bằng công thức Pan-Schreiber
    X0_mat = calculate_X0_pan_schreiber(A_mat)
    
    print_formatted_matrix(X0_mat, "MA TRẬN KHỞI TẠO X0 (Pan-Schreiber)", ".9f")

    inverse_iteration_verbose(
        A_mat, 
        X0_mat, 
        tol=1e-7, 
        max_iter=100,
        val_fmt=".9f", 
        err_fmt=".7e"
    )