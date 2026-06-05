import numpy as np

def newton_raphson_system_verbose(F_func, J_func, X0, tol=1e-5, max_iter=50, val_fmt=".5f", err_fmt=".3e"):
    """
    Giải hệ phương trình phi tuyến n ẩn bằng phương pháp Newton-Raphson.
    """
    X_curr = np.array(X0, dtype=float)
    n = len(X_curr)
    
    print("\n" + "="*85)
    print(f" BẮT ĐẦU PHƯƠNG PHÁP NEWTON-RAPHSON CHO HỆ {n} PHƯƠNG TRÌNH PHI TUYẾN")
    print("="*85)
    print(f"[*] Vector khởi tạo X^(0) = [{', '.join([f'{v:{val_fmt}}' for v in X_curr])}]")
    print("-" * 85)

    for k in range(max_iter):
        print(f"\n>>> BƯỚC LẶP k = {k}")
        
        # 1. Tính Vector F và Ma trận Jacobian
        F_val = np.array(F_func(X_curr), dtype=float)
        J_val = np.array(J_func(X_curr), dtype=float)
        
        # Kiểm tra NaN/Inf để tránh sập chương trình
        if np.any(np.isnan(F_val)) or np.any(np.isinf(F_val)):
            print("[-] CẢNH BÁO: Phân kỳ, giá trị F(X) tiến tới NaN hoặc Vô cực!")
            return None

        # 2. In giá trị trung gian
        print("    Ma trận Jacobian J(X^k):")
        for row in J_val:
            print("      [" + "  ".join([f"{v:{val_fmt}}" for v in row]) + "]")
            
        print(f"    Vector hàm F(X^k): [{', '.join([f'{v:{val_fmt}}' for v in F_val])}]")
        
        # 3. Giải hệ phương trình tuyến tính
        try:
            delta_X = np.linalg.solve(J_val, -F_val)
        except np.linalg.LinAlgError:
            print("[-] CẢNH BÁO: Ma trận Jacobian suy biến (Det = 0). Thuật toán thất bại!")
            return None # Trả về None thay vì X_curr để đánh dấu thất bại
            
        print(f"    Vector gia số Delta_X: [{', '.join([f'{v:{val_fmt}}' for v in delta_X])}]")
        
        # 4. Cập nhật nghiệm mới
        X_next = X_curr + delta_X
        
        # 5. Tính sai số (Chuẩn vô cùng của Delta_X VÀ giá trị của F)
        err_X = np.max(np.abs(delta_X))
        err_F = np.max(np.abs(F_val))
        
        print(f"=> Vector X^({k+1}) = [{', '.join([f'{v:{val_fmt}}' for v in X_next])}]")
        print(f"=> Sai số ||Delta_X||_inf = {err_X:{err_fmt}} | ||F(X)|| = {err_F:{err_fmt}}")

        X_curr = X_next

        # 6. Kiểm tra điều kiện dừng KÉP
        if err_X < tol and err_F < tol:
            print("\n" + "="*85)
            print(f" THUẬT TOÁN HỘI TỤ SAU {k+1} BƯỚC LẶP")
            print(f" KẾT LUẬN NGHIỆM X*: [{', '.join([f'{v:{val_fmt}}' for v in X_curr])}]")
            print("="*85)
            return X_curr

    print(f"\n[-] CẢNH BÁO: Đạt tới số bước lặp tối đa ({max_iter}) nhưng chưa hội tụ!")
    return None

# ==========================================
# KHAI BÁO HỆ PHƯƠNG TRÌNH & CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    def F_system(X):
        x, y = X[0], X[1]
        f1 = x**2 + y**2 - 4.0
        f2 = x**2 - y - 1.0
        return [f1, f2]
        
    def Jacobian_system(X):
        x, y = X[0], X[1]
        return [
            [2*x, 2*y],
            [2*x, -1.0]
        ]

    X0_test = [1.5, 2.0]
    
    result = newton_raphson_system_verbose(
        F_func=F_system, 
        J_func=Jacobian_system, 
        X0=X0_test, 
        tol=1e-5
    )