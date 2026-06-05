import numpy as np

def print_matrix(M, title, val_fmt):
    print(f"\n--- {title} ---")
    for row in M:
        print("[ " + "  ".join([f"{v:{val_fmt}}" for v in row]) + " ]")

def danilevsky_method_verbose(A_input, val_fmt=".7f"):
    """
    Phương pháp Danilevsky đưa ma trận về dạng chuẩn Frobenius 
    để tìm đa thức đặc trưng (Tối ưu bằng NumPy).
    """
    # Chuyển đổi list đầu vào thành mảng NumPy 2 chiều
    A = np.array(A_input, dtype=float)
    n = A.shape[0]
    
    print("\n" + "="*80)
    print(" BẮT ĐẦU PHƯƠNG PHÁP DANILEVSKY TÌM ĐA THỨC ĐẶC TRƯNG")
    print("="*80)
    print_matrix(A, "Ma trận ban đầu A^(0)", val_fmt)

    # Bước lặp đi từ n-2 lùi về 0 (Tương đương từ dòng cuối lên dòng 2 trong lý thuyết)
    for k in range(n - 2, -1, -1):
        target_row = k + 1
        pivot = A[target_row, k] 
        
        print("\n" + "-"*80)
        print(f" BƯỚC {n - 1 - k}: XỬ LÝ DÒNG {target_row + 1} (Đưa về dạng ... 1  0 ...)")
        print("-"*80)
        
        # 1. Kiểm tra phần tử chốt (Pivoting)
        if abs(pivot) < 1e-12:
            print(f"[-] Phần tử chốt a_{target_row+1},{k+1} = 0. Cần tìm cột để hoán vị!")
            swap_j = -1
            for j in range(k - 1, -1, -1):
                if abs(A[target_row, j]) > 1e-12:
                    swap_j = j
                    break
            
            if swap_j == -1:
                print("[-] LỖI: Không tìm thấy phần tử khác 0 để hoán vị. Ma trận có dạng khối.")
                return A
            
            print(f"[*] Đang hoán vị cột {k+1} và {swap_j+1}, sau đó hoán vị dòng {k+1} và {swap_j+1}.")
            # Hoán vị cột và dòng
            A[:, [k, swap_j]] = A[:, [swap_j, k]]
            A[[k, swap_j], :] = A[[swap_j, k], :]
            pivot = A[target_row, k]
            print_matrix(A, "Ma trận sau khi hoán vị", val_fmt)

        # 2. Xây dựng ma trận nghịch đảo M_inv (Lấy nguyên dòng target_row của A)
        M_inv = np.eye(n)
        M_inv[k, :] = A[target_row, :]
        print(" -> M_inv: Tạo bằng cách thay dòng {} của ma trận đơn vị bằng dòng {} của A".format(k+1, target_row+1))
        print_matrix(M_inv, f"Ma trận M_inv_{target_row}", val_fmt)

        # 3. Xây dựng ma trận M
        M = np.eye(n)
        M[k, :] = -A[target_row, :] / pivot
        M[k, k] = 1.0 / pivot
        print(" -> M: Tính bằng công thức m_{ij} = -a_{ij}/a_{pivot} và m_{ii} = 1/a_{pivot}")
        print_matrix(M, f"Ma trận M_{target_row}", val_fmt)

        # 4. Thực hiện phép biến đổi đồng dạng A = M * A * M_inv

        A = M_inv @ A @ M        
        # Nếu là bước cuối cùng, gán tên là Ma trận Frobenius
        if k == 0:
            print_matrix(A, "MA TRẬN DẠNG CHUẨN FROBENIUS (F)", val_fmt)
            print(" -> ĐÂY CHÍNH LÀ MA TRẬN FROBENIUS! Dòng đầu tiên chứa các hệ số của đa thức đặc trưng.")
        else:
            print_matrix(A, f"Ma trận A^({n - 1 - k}) = M * A * M_inv", val_fmt)

    # 5. Đọc kết quả Đa thức đặc trưng
    print("\n" + "="*80)
    print(" KẾT QUẢ ĐA THỨC ĐẶC TRƯNG")
    print("="*80)
    
    # Dòng đầu tiên của ma trận Frobenius chứa p_1, p_2, ..., p_n
    p_coeffs = A[0]
    print("Các hệ số đọc từ dòng 1 của ma trận Frobenius: ", [round(p, 4) for p in p_coeffs])
    
    # Xây dựng chuỗi phương trình
    terms = [f"lambda^{n}"]
    for i, p in enumerate(p_coeffs):
        power = n - 1 - i
        # Hệ số trong công thức là trừ đi p_i, nên ta đảo dấu để hiển thị
        sign = "-" if p > 0 else "+"
        term = f"{sign} {abs(p):{val_fmt}}"
        
        if power > 0:
            term += f"*lambda^{power}" if power > 1 else "*lambda"
        terms.append(term)
        
    poly_str = " ".join(terms)
    
    # Xét dấu (-1)^n của định thức tổng quát
    det_sign = "-" if n % 2 != 0 else ""
    
    print(f"\nĐa thức đặc trưng P(lambda) = det(A - lambda*I):")
    print(f"P(lambda) = {det_sign}({poly_str})")
    print("\nPhương trình đặc trưng để tìm giá trị riêng:")
    print(f"{poly_str} = 0")
    
    return A

# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    A_matrix = [
        [1.0, 2.0, 3.0,4],
        [2.0, 1,3,2],
        [3.0,2,1,2],
        [4,3,2,1]
    ]
    
    danilevsky_method_verbose(A_matrix, val_fmt=".7f")