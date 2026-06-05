import math

def print_matrix(M, title, val_fmt=".4f"):
    """Hàm phụ trợ để in ma trận với format tùy chỉnh."""
    print(f"\n--- {title} ---")
    for row in M:
        print("[ " + "  ".join([f"{val:{val_fmt}}" for val in row]) + " ]")

def lu_solve_verbose_detailed(A, B, val_fmt=".4f"):
    """
    Giải hệ Ax = B bằng phân rã LU (Doolittle).
    CÓ IN RA CÁC BƯỚC TRUNG GIAN.
    """
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0 for j in range(n)] for i in range(n)]

    print("\n" + "="*60)
    print(" BẮT ĐẦU PHÂN RÃ LU (DOOLITTLE) VÀ GIẢI HỆ")
    print("="*60)
    
    # 1. Quá trình phân rã LU (Giữ nguyên như trước)
    for i in range(n):
        for j in range(i, n):
            sum_lu = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_lu
            
        for j in range(i + 1, n):
            sum_lu = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - sum_lu) / U[i][i]
            
    print_matrix(L, "Ma trận tam giác dưới L (Lower)", val_fmt)
    print_matrix(U, "Ma trận tam giác trên U (Upper)", val_fmt)

    # ---------------------------------------------------------
    # 2. GIẢI HỆ THẾ THUẬN (FORWARD SUBSTITUTION): Ly = B
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(">>> BƯỚC 2: THẾ THUẬN GIẢI HỆ Ly = B (Từ trên xuống dưới)")
    print("-"*60)
    y = [0.0] * n
    for i in range(n):
        # Trình bày phương trình hiện tại
        eq_str = " + ".join([f"({L[i][j]:{val_fmt}} * y_{j})" for j in range(i)])
        if eq_str:
            eq_str = f"{eq_str} + "
        print(f"[*] Phương trình hàng {i}: {eq_str}({L[i][i]:{val_fmt}} * y_{i}) = {B[i]:{val_fmt}}")
        
        # Tính toán
        sum_Ly = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (B[i] - sum_Ly) / L[i][i] # Trong Doolittle L[i][i] luôn = 1
        
        print(f"    -> Thế các y đã biết vào: {sum_Ly:{val_fmt}} + ({L[i][i]:{val_fmt}} * y_{i}) = {B[i]:{val_fmt}}")
        print(f"    -> y_{i} = {y[i]:{val_fmt}}\n")

    # ---------------------------------------------------------
    # 3. GIẢI HỆ THẾ NGƯỢC (BACKWARD SUBSTITUTION): Ux = y
    # ---------------------------------------------------------
    print("-"*60)
    print(">>> BƯỚC 3: THẾ NGƯỢC GIẢI HỆ Ux = y (Từ dưới lên trên)")
    print("-"*60)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        # Trình bày phương trình hiện tại
        eq_str = " + ".join([f"({U[i][j]:{val_fmt}} * x_{j})" for j in range(i+1, n)])
        prefix = f"({U[i][i]:{val_fmt}} * x_{i})"
        full_eq = f"{prefix} + {eq_str}" if eq_str else prefix
        print(f"[*] Phương trình hàng {i}: {full_eq} = y_{i} ({y[i]:{val_fmt}})")
        
        # Tính toán
        sum_Ux = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_Ux) / U[i][i]
        
        print(f"    -> Thế các x đã biết vào: ({U[i][i]:{val_fmt}} * x_{i}) + {sum_Ux:{val_fmt}} = {y[i]:{val_fmt}}")
        print(f"    -> x_{i} = ({y[i]:{val_fmt}} - {sum_Ux:{val_fmt}}) / {U[i][i]:{val_fmt}} = {x[i]:{val_fmt}}\n")

    print("=> KẾT LUẬN NGHIỆM TỔNG QUÁT x: [" + ", ".join([f"{val:{val_fmt}}" for val in x]) + "]")
    return x

def cholesky_solve_verbose_detailed(A, B, val_fmt=".4f"):
    """
    Giải hệ Ax = B bằng phân rã Cholesky (LL^T).
    Có in ra các bước thế thuận, thế ngược chi tiết như giải tay.
    """
    n = len(A)
    # Kiểm tra tính đối xứng của ma trận A
    for i in range(n):
        for j in range(i):
            if abs(A[i][j] - A[j][i]) > 1e-9:
                raise ValueError("Ma trận A không đối xứng. Không thể dùng phương pháp Cholesky.")

    L = [[0.0 for j in range(n)] for i in range(n)]
    
    print("\n" + "="*60)
    print(" BẮT ĐẦU PHÂN RÃ CHOLESKY (LL^T) VÀ GIẢI HỆ")
    print("="*60)

    # ---------------------------------------------------------
    # 1. QUÁ TRÌNH PHÂN RÃ MA TRẬN A = LL^T
    # ---------------------------------------------------------
    print("\n>>> BƯỚC 1: TÍNH TOÁN MA TRẬN TAM GIÁC DƯỚI L")
    for i in range(n):
        for j in range(i + 1):
            sum_L = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j: # Phần tử trên đường chéo chính
                val_inside_sqrt = A[i][i] - sum_L
                if val_inside_sqrt <= 0:
                    raise ValueError(f"Ma trận không xác định dương (L[{i}][{i}]^2 <= 0).")
                L[i][i] = math.sqrt(val_inside_sqrt)
            else: # Phần tử dưới đường chéo chính
                L[i][j] = (A[i][j] - sum_L) / L[j][j]

    print_matrix(L, "Ma trận tam giác dưới L", val_fmt)

    # Tạo ma trận chuyển vị L^T
    L_T = [[L[j][i] for j in range(n)] for i in range(n)]
    print_matrix(L_T, "Ma trận tam giác trên L^T (Chuyển vị của L)", val_fmt)

    # ---------------------------------------------------------
    # 2. GIẢI HỆ THẾ THUẬN (FORWARD SUBSTITUTION): Ly = B
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    print(">>> BƯỚC 2: THẾ THUẬN GIẢI HỆ Ly = B (Từ trên xuống dưới)")
    print("-"*60)
    y = [0.0] * n
    for i in range(n):
        # Trình bày phương trình hiện tại
        eq_str = " + ".join([f"({L[i][j]:{val_fmt}} * y_{j})" for j in range(i)])
        if eq_str:
            eq_str = f"{eq_str} + "
        print(f"[*] Phương trình hàng {i}: {eq_str}({L[i][i]:{val_fmt}} * y_{i}) = {B[i]:{val_fmt}}")
        
        # Tính toán
        sum_Ly = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (B[i] - sum_Ly) / L[i][i]
        
        print(f"    -> Thế các y đã biết vào: {sum_Ly:{val_fmt}} + ({L[i][i]:{val_fmt}} * y_{i}) = {B[i]:{val_fmt}}")
        print(f"    -> y_{i} = {y[i]:{val_fmt}}\n")

    # ---------------------------------------------------------
    # 3. GIẢI HỆ THẾ NGƯỢC (BACKWARD SUBSTITUTION): L^Tx = y
    # ---------------------------------------------------------
    print("-" * 60)
    print(">>> BƯỚC 3: THẾ NGƯỢC GIẢI HỆ L^Tx = y (Từ dưới lên trên)")
    print("-" * 60)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        # Trình bày phương trình hiện tại
        eq_str = " + ".join([f"({L_T[i][j]:{val_fmt}} * x_{j})" for j in range(i+1, n)])
        prefix = f"({L_T[i][i]:{val_fmt}} * x_{i})"
        full_eq = f"{prefix} + {eq_str}" if eq_str else prefix
        print(f"[*] Phương trình hàng {i}: {full_eq} = y_{i} ({y[i]:{val_fmt}})")
        
        # Tính toán
        sum_LTx = sum(L_T[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_LTx) / L_T[i][i]
        
        print(f"    -> Thế các x đã biết vào: ({L_T[i][i]:{val_fmt}} * x_{i}) + {sum_LTx:{val_fmt}} = {y[i]:{val_fmt}}")
        print(f"    -> x_{i} = ({y[i]:{val_fmt}} - {sum_LTx:{val_fmt}}) / {L_T[i][i]:{val_fmt}} = {x[i]:{val_fmt}}\n")

    print("=> KẾT LUẬN NGHIỆM TỔNG QUÁT x: [" + ", ".join([f"{val:{val_fmt}}" for val in x]) + "]")
    return x

# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Ma trận A phải đối xứng và xác định dương
    A_chol = [
        [  4.0,  12.0, -16.0],
        [ 12.0,  37.0, -43.0],
        [-16.0, -43.0,  98.0]
    ]
    B_chol = [4.0, 1.0, 10.0]
    
    # Bạn có thể điều chỉnh format ở đây
    my_format = ">8.7f"
    
    # Chạy thử phân rã LU
    try:
        lu_solve_verbose_detailed(A_chol, B_chol, val_fmt=my_format)
    except ValueError as e:
        print(e)
        
    # Chạy thử phân rã Cholesky
    try:
        cholesky_solve_verbose_detailed(A_chol, B_chol, val_fmt=my_format)
    except ValueError as e:
        print(e)