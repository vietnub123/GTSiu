import numpy as np

def eigenvalue_qr_algorithm_step_by_step(A, max_iter=1000, tol=1e-10, val_fmt=".5f", max_print_steps=4):
    """
    Thuật toán QR lặp để tìm toàn bộ phổ trị riêng của ma trận vuông A.
    Quá trình phân rã được kết xuất tường minh từng phép tính ma trận.
    """
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Phân tích trị riêng chỉ áp dụng cho không gian vector vuông (ma trận vuông).")

    A_curr = A.astype(float).copy()
    
    print("\n" + "="*80)
    print(" BẮT ĐẦU PHÂN TÍCH TRỊ RIÊNG BẰNG THUẬT TOÁN QR (QUY TRÌNH THỦ CÔNG)")
    print("="*80)

    for i in range(max_iter):
        # 1. Phân rã trực giao không gian cột của A_curr
        Q, R = np.linalg.qr(A_curr)
        
        # 2. Phép biến đổi đồng dạng bảo toàn phổ trị riêng
        A_next = R @ Q
        
        # 3. Kết xuất tường minh các ma trận thành phần
        if i < max_print_steps:
            print(f"\n--- BƯỚC LẶP k = {i+1} ---")
            print("1. Ma trận hiện tại A_k:")
            print(np.round(A_curr, 5))
            print("\n2. Ma trận trực giao Q_k (Orthogonal Matrix):")
            print(np.round(Q, 5))
            print("\n3. Ma trận tam giác trên R_k (Upper Triangular Matrix):")
            print(np.round(R, 5))
            print("\n4. Ma trận chiếu chu kỳ sau A_{k+1} = R_k * Q_k:")
            print(np.round(A_next, 5))
            print("-" * 40)
        elif i == max_print_steps:
            print(f"\n[...] Quá trình tiếp tục tiệm cận. Ẩn chi tiết để tối ưu hóa không gian hiển thị.")
        
        # 4. Đánh giá sai số hội tụ (chuẩn Frobenius của phần dư tam giác dưới)
        lower_triangular_norm = np.linalg.norm(np.tril(A_next, k=-1))
        
        if lower_triangular_norm < tol:
            print(f"\n[*] HỆ THỐNG HỘI TỤ DẠNG SCHUR SAU {i+1} BƯỚC LẶP.")
            A_curr = A_next
            break
            
        A_curr = A_next

    if lower_triangular_norm >= tol:
        print("\n[-] Cảnh báo: Đạt giới hạn lặp tối đa nhưng hệ thống chưa phân rã hoàn toàn.")

    # Tách chiết trị riêng từ đường chéo chính của ma trận Schur
    eigenvalues = np.diag(A_curr)
    
    return eigenvalues, A_curr

# ==========================================
# GIAO THỨC CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    # Khởi tạo ma trận đầy đủ (dense matrix) để quan sát rõ quá trình hoán vị và trực giao hóa
    A_square = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.0, 3.0]
    ])
    
    print("MA TRẬN KHỞI TẠO A_0:")
    print(A_square)

    # Kích hoạt thuật toán với giới hạn hiển thị 4 bước lặp đầu tiên
    eigenvalues, Schur_form = eigenvalue_qr_algorithm_step_by_step(A_square, max_print_steps=4)
    
    print("\n" + "="*80)
    print(" KẾT QUẢ PHÂN TÍCH PHỔ TRỊ RIÊNG (SPECTRUM ANALYSIS)")
    print("="*80)
    print("1. Ma trận dạng Schur:\n", np.round(Schur_form, 5))
    print("\n2. Phân bố phổ trị riêng (Eigenvalues):")
    
    # Sắp xếp trị riêng theo thứ tự giảm dần để chuẩn hóa đầu ra
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    for idx, val in enumerate(sorted_eigenvalues):
        print(f"   lambda_{idx+1} = {val:.5f}")
        
    # Kiểm chứng độ chính xác thông qua hàm tính toán phổ trị riêng từ thư viện lõi
    eigenvalues_exact = np.linalg.eigvals(A_square)
    sorted_exact = np.sort(eigenvalues_exact)[::-1]
    print("\n=> Đối chiếu với nghiệm thư viện chuẩn (np.linalg.eigvals):")
    for idx, val in enumerate(sorted_exact):
         print(f"   lambda_exact_{idx+1} = {val:.5f}")