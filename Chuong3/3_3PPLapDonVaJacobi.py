def check_diagonally_dominant(A):
    """
    Kiểm tra ma trận A có chéo trội nghiêm ngặt hàng hay không.
    Trả về (True/False, hệ số co q)
    """
    n = len(A)
    q_max = 0.0
    is_dominant = True
    
    for i in range(n):
        diag_val = abs(A[i][i])
        sum_row = sum(abs(A[i][j]) for j in range(n) if j != i)
        
        if diag_val == 0:
            return False, float('inf')
            
        q_i = sum_row / diag_val
        if q_i > q_max:
            q_max = q_i
            
        if sum_row >= diag_val:
            is_dominant = False
            
    return is_dominant, q_max


def jacobi_solve_verbose(A, B, x0=None, tol=1e-4, max_iter=50, val_fmt=".4f", err_fmt=".6f"):
    """
    Giải hệ Ax = B bằng phương pháp lặp Jacobi.
    val_fmt: Định dạng in cho các giá trị ẩn x, hệ số (VD: ".4f").
    err_fmt: Định dạng in riêng cho sai số (VD: ".6f", ".2e").
    """
    n = len(A)
    x_curr = [0.0] * n if x0 is None else list(x0)
        
    print("\n" + "="*75)
    print(" BẮT ĐẦU PHƯƠNG PHÁP LẶP JACOBI")
    print("="*75)
    
    # Kiểm tra tính chéo trội
    is_dominant, q = check_diagonally_dominant(A)
    if is_dominant:
        print(f"[+] Ma trận chéo trội nghiêm ngặt. Hệ số co q = {q:{val_fmt}} < 1. Chắc chắn hội tụ.")
    else:
        print(f"[-] CẢNH BÁO: Ma trận KHÔNG chéo trội nghiêm ngặt (q = {q:{val_fmt}}). Có thể không hội tụ!")

    # In công thức lặp tổng quát
    print("\n>>> CÔNG THỨC LẶP RÚT GỌN CHO TỪNG ẨN:")
    for i in range(n):
        eq_parts = [f"({-A[i][j]:{val_fmt}} * x_{j})" for j in range(n) if j != i]
        print(f"    x_{i}^(k+1) = (1/{A[i][i]:{val_fmt}}) * [ {B[i]:{val_fmt}} + {' + '.join(eq_parts)} ]")

    print("\n" + "-"*75)
    print(f"[*] Vector khởi tạo x^(0) = [{', '.join([f'{v:{val_fmt}}' for v in x_curr])}]")
    print("-" * 75)

    # Vòng lặp Jacobi
    for k in range(1, max_iter + 1):
        print(f"\n>>> BƯỚC LẶP k = {k} <<<")
        x_next = [0.0] * n
        
        for i in range(n):
            sum_Ax = 0.0
            calc_str_parts = []
            
            for j in range(n):
                if i != j:
                    # Jacobi luôn dùng x_curr (vector của bước lặp trước)
                    val = -A[i][j] * x_curr[j]
                    sum_Ax += val
                    calc_str_parts.append(f"({-A[i][j]:{val_fmt}} * {x_curr[j]:{val_fmt}})")
            
            x_next[i] = (B[i] + sum_Ax) / A[i][i]
            calc_str = " + ".join(calc_str_parts)
            print(f"[*] x_{i}^({k}) = (1/{A[i][i]:{val_fmt}}) * [ {B[i]:{val_fmt}} + {calc_str} ] = {x_next[i]:{val_fmt}}")

        # Đánh giá sai số theo chuẩn vô cùng
        error = max(abs(x_next[i] - x_curr[i]) for i in range(n))
        print(f"=> Vector x^({k}) = [{', '.join([f'{v:{val_fmt}}' for v in x_next])}]")
        print(f"=> Sai số ||x^({k}) - x^({k-1})||_inf = {error:{err_fmt}}")

        # Cập nhật vector cho bước lặp tiếp theo
        x_curr = x_next[:]

        if error < tol:
            print("\n" + "="*75)
            print(f" THUẬT TOÁN HỘI TỤ SAU {k} BƯỚC LẶP (Sai số < {tol:{err_fmt}})")
            print(f" KẾT LUẬN NGHIỆM: [{', '.join([f'{v:{val_fmt}}' for v in x_curr])}]")
            print("="*75)
            return x_curr

    print("\nCẢNH BÁO: Đạt tới số bước lặp tối đa nhưng chưa hội tụ!")
    return x_curr

# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    A_jacobi = [
        [ 10.0, -1.0,  2.0],
        [-1.0,  11.0, -1.0],
        [ 2.0,  -1.0, 10.0]
    ]
    B_jacobi = [6.0, 25.0, -11.0]
    
    # Chạy lặp Jacobi với định dạng tùy chỉnh
    jacobi_solve_verbose(
        A_jacobi, 
        B_jacobi, 
        x0=[0.0, 0.0, 0.0], 
        tol=1e-5, 
        max_iter=15, 
        val_fmt=".5f",   # In giá trị với 5 chữ số thập phân
        err_fmt=".3e"    # In sai số dưới dạng khoa học (VD: 1.234e-03)
    )