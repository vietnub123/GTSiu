import numpy as np

def svd_pseudocode_verbose(A, k_components=None, tol=1e-5, max_iter=100, val_fmt=".7f", err_fmt=".7e"):
    """
    Thuật toán SVD xấp xỉ áp dụng phương pháp Lũy thừa trên ma trận Gram (M = A^T A)
    và kỹ thuật Xuống thang (Deflation), tuân thủ nghiêm ngặt cấu trúc mã giả đại số.
    """
    m, n = A.shape
    if k_components is None:
        k_components = min(m, n)
        
    # Khởi tạo A_curr
    A_curr = A.astype(float).copy()
    
    U_cols = []
    Sigmas = []
    V_rows = []
    
    print("\n" + "="*80)
    print(" BẮT ĐẦU PHÂN TÍCH GIÁ TRỊ KỲ DỊ (SVD) THEO CẤU TRÚC MÃ GIẢ")
    print("="*80)
    
    for i in range(k_components):
        print(f"\n" + "-"*60)
        print(f" TÌM THÀNH PHẦN KỲ DỊ THỨ {i+1}")
        print("-" * 60)
        
        # 1. Tính ma trận trung gian M = A^T * A
        M = A_curr.T @ A_curr
        
        # 2. Khởi tạo vector ngẫu nhiên x_0 trong không gian R^n
        # (Sử dụng seed cố định nội bộ để đảm bảo tính tái lập trong các bài kiểm tra số học)
        np.random.seed(42 + i)
        x_curr = np.random.rand(n)
        
        print(f" [*] Vector khởi tạo ngẫu nhiên x_0 = [{', '.join([f'{val:{val_fmt}}' for val in x_curr])}]")
        
        step = 1

        while step <= max_iter:
            # 3. Tính tích vector y = M * x_curr
            y = M @ x_curr
            
# 4. Xác định hệ số chuẩn hóa c theo các chuẩn (Norms) không gian:
            # -----------------------------------------------------------------
            # [Mặc định] Chuẩn vô cực (L_infinity norm): Tiệm cận phần tử có module lớn nhất.
            c = np.max(np.abs(y)) 
            
            # [Lựa chọn thay thế 1] Chuẩn L1 (Manhattan/Taxicab norm): 
            # Sử dụng khi: Cần chuẩn hóa theo tổng trị tuyệt đối của hệ tọa độ.
            # Cú pháp: c = np.sum(np.abs(y))
            
            # [Lựa chọn thay thế 2] Chuẩn L2 (Euclidean norm): 
            # Sử dụng khi: Yêu cầu chuẩn hóa bảo toàn năng lượng vector về độ dài 1.
            # Cú pháp: c = np.linalg.norm(y, ord=2)
            # -----------------------------------------------------------------
            
            # Điều kiện giới hạn phổ suy biến
            if c < 1e-15:
                break
                
            # 5. Chuẩn hóa vector x_next
            x_next = y / c
            
            # 6. Đánh giá sai số hội tụ dựa trên Metric không gian
            # -----------------------------------------------------------------
            # Hàm np.linalg.norm của NumPy mặc định sử dụng tham số ord=None (tương đương chuẩn L2 cho vector 1D).
            # Để thay đổi độ đo sai số, cần khai báo minh bạch tham số `ord` trong mọi hàm norm:
            # - ord=1      : Đánh giá sai số theo chuẩn L1.
            # - ord=2      : Đánh giá sai số theo chuẩn L2.
            # - ord=np.inf : Đánh giá sai số theo chuẩn L_infinity.
            
            # Ví dụ dưới đây đang khai báo tường minh ord=2 (Chuẩn L2). 
            # Nếu đề bài yêu cầu chuẩn khác, thay đổi toàn bộ giá trị của tham số `ord` tương ứng.
            error = min(np.linalg.norm(x_next - x_curr, ord=2), 
                        np.linalg.norm(x_next + x_curr, ord=2))

            # Hiệu chỉnh ánh xạ ngược chiều
            # LƯU Ý BẮT BUỘC: Tham số `ord` tại đây phải đồng nhất tuyệt đối với tham số `ord` trong phép tính `error` phía trên để đảm bảo tính nhất quán của không gian định chuẩn.
            if np.linalg.norm(x_next + x_curr, ord=2) < np.linalg.norm(x_next - x_curr, ord=2):
                x_next = -x_next # Hiệu chỉnh chiều vector đối xứng
            
            # Xuất chi tiết trạng thái vòng lặp
            print(f"\n  >>> BƯỚC LẶP step = {step}")
            print(f"      Vector y = [{', '.join([f'{val:{val_fmt}}' for val in y])}]")
            print(f"      Hệ số c (max|y|) = {c:{val_fmt}}")
            print(f"      Vector x_next = [{', '.join([f'{val:{val_fmt}}' for val in x_next])}]")
            print(f"      => Sai số = {error:{err_fmt}}")
            
            # 7. Kiểm tra điều kiện hội tụ
            if error < tol:
                print(f"\n  => HỘI TỤ KHÔNG GIAN RIÊNG SAU {step} BƯỚC LẶP")
                x_curr = x_next
                break
            
            # Cập nhật trạng thái vector
            x_curr = x_next
            step += 1
            
        if step > max_iter:
            print("\n  [-] Đạt giới hạn lặp tối đa nhưng hệ thống chưa tiệm cận độ chính xác mong muốn!")
            
        # Xử lý hậu hội tụ
        if c < 1e-15:
            print(f"\n[*] Giá trị kỳ dị xấp xỉ bằng 0. Hệ ma trận đã được triệt tiêu hoàn toàn!")
            break
            
        # 8. Tính toán các thành phần SVD
        lambda_i = c
        sigma_i = np.sqrt(lambda_i)
        
        # Chuẩn hóa vector kỳ dị phải v_i về định mức độ dài L2 = 1
        v_i = x_curr / np.linalg.norm(x_curr)
        
        # Phân rã vector kỳ dị trái u_i
        u_i = (A_curr @ v_i) / sigma_i
        
        # Lưu trữ các không gian con
        U_cols.append(u_i)
        Sigmas.append(sigma_i)
        V_rows.append(v_i)
        
        print(f"\n  [+] Thông số thành phần kỳ dị thứ {i+1}:")
        print(f"      sigma_{i+1} = {sigma_i:{val_fmt}}")
        print(f"      u_{i+1} = [{', '.join([f'{val:{val_fmt}}' for val in u_i])}]")
        print(f"      v_{i+1} = [{', '.join([f'{val:{val_fmt}}' for val in v_i])}]")
        
        # 9. Phương pháp Xuống thang (Hotelling Deflation)
        A_curr = A_curr - sigma_i * np.outer(u_i, v_i)
        
        print(f"\n[*] Ma trận A sau khi xuống thang bước {i+1} (Chuẩn phần dư = {np.linalg.norm(A_curr):{val_fmt}}):")
        for row in A_curr:
            print("    [" + "  ".join([f"{val:{val_fmt}}" for val in row]) + "]")

    # Xây dựng các khối ma trận cấu thành
    U = np.column_stack(U_cols)
    S = np.diag(Sigmas)
    Vt = np.vstack(V_rows)
    
    return U, S, Vt

# ==========================================
# GIAO THỨC CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Cấu hình môi trường in ấn số liệu
    np.set_printoptions(precision=7, suppress=True)

    # Khởi tạo ma trận đánh giá
    A_rect = np.array([
        [3.0, 1.0, 2.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -0.5]
    ])
    
    print("MA TRẬN KHỞI TẠO A:")
    print(A_rect)

    # Kích hoạt chuỗi phân tích
    U, Sigma, Vt = svd_pseudocode_verbose(A_rect, k_components=3, tol=1e-6)
    
    print("\n" + "="*80)
    print(" CẤU TRÚC PHÂN RÃ HÌNH THỨC (A ≈ U * Sigma * V^T)")
    print("="*80)
    print("1. Ma trận U:\n", U)
    print("\n2. Ma trận Sigma:\n", Sigma)
    print("\n3. Ma trận V^T:\n", Vt)
    
    # Tái thiết lập ma trận từ không gian con
    A_reconstruct = U @ Sigma @ Vt
    print("\nMa trận A phục hồi từ phương trình SVD:\n", A_reconstruct)
    
    # Kiểm chứng chuẩn sai số phục hồi L1
    print(f"=> Sai số phục hồi tuyệt đối: {np.linalg.norm(A_rect - A_reconstruct, 1):.7e}")