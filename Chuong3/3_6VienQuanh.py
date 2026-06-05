import numpy as np

def print_matrix(M, title, val_fmt):
    print(f"\n--- {title} ---")
    for row in M:
        print("[ " + "  ".join([f"{v:{val_fmt}}" for v in row]) + " ]")

def bordering_inverse_verbose(A_input, val_fmt=".4f", tol=1e-12):
    """
    Tìm ma trận nghịch đảo bằng phương pháp viền quanh (Bordering Method).
    Tối ưu hóa bằng NumPy (Block Matrix Operations).
    """
    A = np.array(A_input, dtype=float)
    
    # [BỔ SUNG 1]: Kiểm tra ma trận vuông
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("LỖI: Ma trận đầu vào bắt buộc phải là ma trận vuông.")
        
    n = A.shape[0]
    
    # Bước khởi đầu: Ma trận cấp 1
    if abs(A[0, 0]) < tol:
        raise ValueError("LỖI: Phần tử a11 ≈ 0. Vui lòng hoán vị dòng/cột ma trận trước khi đưa vào hàm.")
    
    inv_A_k = np.array([[1.0 / A[0, 0]]])
    
    print("\n" + "="*80)
    print(" BẮT ĐẦU PHƯƠNG PHÁP VIỀN QUANH (NUMPY OPTIMIZED)")
    print("="*80)
    print_matrix(inv_A_k, "Nghịch đảo ma trận cấp 1 (A1^-1)", val_fmt)

    # Lặp để tăng dần cấp từ 2 lên n
    for k in range(1, n):
        print(f"\n>>> Xây dựng nghịch đảo cấp {k+1} từ cấp {k}")
        
        # 1. Trích xuất các thành phần khối siêu tốc bằng Slicing
        u = A[:k, k]
        v_T = A[k, :k]
        a_kk = A[k, k]

        # 2. Tính các đại lượng trung gian bằng toán tử @
        inv_Ak_u = inv_A_k @ u
        vT_inv_Ak = v_T @ inv_A_k
        vT_inv_Ak_u = v_T @ inv_Ak_u 
        
        # 3. Tính hệ số alpha (Schur complement inverse)
        denom = a_kk - vT_inv_Ak_u
        if abs(denom) < tol:  # [BỔ SUNG 2]: Dùng tol thay vì fix cứng 1e-12
            raise ValueError(f"LỖI: Mẫu số ≈ 0 tại cấp {k+1}. Ma trận con suy biến, cần hoán vị ma trận gốc.")
            
        alpha = 1.0 / denom
        print(f"[*] Tính alpha_{k+1} = 1 / ({a_kk:{val_fmt}} - {vT_inv_Ak_u:{val_fmt}}) = {alpha:{val_fmt}}")

        # 4. Tính vector cột q và vector dòng r_T
        q = -alpha * inv_Ak_u
        r_T = -alpha * vT_inv_Ak

        # 5. [BỔ SUNG 3]: Tính khối ma trận P tối ưu hơn bằng cách tái sử dụng q và r_T
        # Công thức: P = inv_A_k + (1/alpha) * q * r^T
        P = inv_A_k + (1.0 / alpha) * np.outer(q, r_T)

        # 6. Ghép các khối thành ma trận nghịch đảo cấp k+1
        inv_A_k = np.block([
            [P,            q[:, None]],
            [r_T[None, :], np.array([[alpha]])]
        ])
        
        print_matrix(inv_A_k, f"Ma trận nghịch đảo cấp {k+1} (A{k+1}^-1)", val_fmt)

    return inv_A_k

# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Test case của bạn: A_test @ A_test tạo ra ma trận vuông dương xác định, 
    # đảm bảo mọi ma trận con đều khả nghịch, không bao giờ bị dính lỗi chia cho 0.
    A_test = [
        [0,-5,-8,-5,-1,-4],
        [10,0,-7,9,-3,5],
        [-3,4,-5,-3,7,5],
        [2,8,7,-6,2,-3],
        [-6,10,-5,-5,1,1],
        [5,1,7,2,9,-9]
    ]
    A_test = np.array(A_test)
    try:
        # Nhân ma trận với chính nó để test
        bordering_inverse_verbose(A_test, val_fmt=".4f")
        
    except ValueError as e:
        print(e)