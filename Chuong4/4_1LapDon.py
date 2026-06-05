import math

def fixed_point_iteration_n_vars(phi_func, X0, tol=1e-5, max_iter=50, val_fmt=".5f", err_fmt=".3e"):
    """
    Giải hệ n phương trình phi tuyến bằng phương pháp lặp đơn.
    Thuật toán tự động nhận diện số lượng ẩn dựa vào vector khởi tạo X0.
    """
    n = len(X0)
    X_curr = list(X0)
    
    print("\n" + "="*80)
    print(f" BẮT ĐẦU LẶP ĐƠN CHO HỆ {n} PHƯƠNG TRÌNH PHI TUYẾN: X = Phi(X)")
    print("="*80)
    print(f"[*] Vector khởi tạo X^(0) = [{', '.join([f'{v:{val_fmt}}' for v in X_curr])}]")
    print("-" * 80)

    for k in range(1, max_iter + 1):
        print(f"\n>>> BƯỚC LẶP k = {k}")
        
        # 1. Tính toán vector mới với cơ chế bắt lỗi an toàn
        try:
            X_next = phi_func(X_curr)
            
            # Kiểm tra xem hàm Phi có trả về đủ số lượng phương trình không
            if len(X_next) != n:
                raise ValueError(f"Hàm Phi trả về {len(X_next)} giá trị, nhưng hệ có {n} ẩn!")
                
        except Exception as e:
            print(f"[-] LỖI TÍNH TOÁN HÀM PHI tại bước {k}: {e}")
            return X_curr
            
        # 2. In chi tiết từng ẩn
        for i in range(n):
            print(f"    x_{i+1}^({k}) = {X_next[i]:{val_fmt}}")
            
        # 3. Đánh giá sai số (Chuẩn vô cùng)
        error = max(abs(X_next[i] - X_curr[i]) for i in range(n))
        
        print(f"=> Vector X^({k}) = [{', '.join([f'{v:{val_fmt}}' for v in X_next])}]")
        print(f"=> Sai số ||X^({k}) - X^({k-1})||_inf = {error:{err_fmt}}")

        # 4. Cập nhật vector hiện tại
        X_curr = list(X_next)

        # 5. Kiểm tra điều kiện dừng
        if error < tol:
            print("\n" + "="*80)
            print(f" THUẬT TOÁN HỘI TỤ SAU {k} BƯỚC LẶP (Sai số < {tol:{err_fmt}})")
            print(f" KẾT LUẬN NGHIỆM X*: [{', '.join([f'{v:{val_fmt}}' for v in X_curr])}]")
            print("="*80)
            return X_curr

    print(f"\n[-] CẢNH BÁO: Đạt tới số bước lặp tối đa ({max_iter}) nhưng chưa hội tụ!")
    return X_curr

# ==========================================
# CÁCH KHAI BÁO HỆ NHIỀU PHƯƠNG TRÌNH (VÍ DỤ)
# ==========================================
if __name__ == "__main__":
    
    # ----------------------------------------------------
    # VÍ DỤ 1: Hệ 3 phương trình riêng biệt (Khai báo thủ công)
    # x1 = (1/3)*cos(x2 * x3) + 0.1
    # x2 = (1/9)*sqrt(x1^2 + 0.1) + 0.2
    # x3 = -(1/10)*exp(-x1 * x2) + 0.3
    # ----------------------------------------------------
    def system_3_vars(X):
        x1, x2, x3 = X[0], X[1], X[2]
        
        nx1 = (1.0/3.0) * math.cos(x2 * x3) + 0.1
        nx2 = (1.0/9.0) * math.sqrt(x1**2 + 0.1) + 0.2
        nx3 = -0.1 * math.exp(-x1 * x2) + 0.3
        
        return [nx1, nx2, nx3]

    print("\n[CHẠY VÍ DỤ 1: Hệ 3 phương trình]")
    fixed_point_iteration_n_vars(
        phi_func=system_3_vars, 
        X0=[0.0, 0.0, 0.0], # Thuật toán tự hiểu n = 3
        tol=1e-5
    )

    # ----------------------------------------------------
    # VÍ DỤ 2: Hệ n phương trình tổng quát (Dùng vòng lặp)
    # Phù hợp cho n = 10, 50, 100...
    # Phương trình quy luật: x_i = (1/2n) * sum(sin(x_j)) + b_i
