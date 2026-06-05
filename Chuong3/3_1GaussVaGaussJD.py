import copy

def print_matrix(Ab, title, val_fmt=".4f"):
    """Hàm phụ trợ để in ma trận mở rộng [A|B] một cách căn chỉnh."""
    print(f"\n--- {title} ---")
    for row in Ab:
        # In các hệ số của A
        str_A = "  ".join([f"{val:{val_fmt}}" for val in row[:-1]])
        # In vế phải B
        str_B = f" | {row[-1]:{val_fmt}}"
        print(f"[{str_A}{str_B} ]")
    print("-" * 40)


def gauss_solve_verbose(A, B, val_fmt=".4f"):
    """
    Giải hệ Ax = B bằng phương pháp Gauss (Khử thuận + Thế ngược).
    CÓ IN RA CÁC BƯỚC TRUNG GIAN.
    """
    n = len(A)
    Ab = [row[:] + [B[i]] for i, row in enumerate(A)]
    
    print("\n" + "="*50)
    print(" BẮT ĐẦU PHƯƠNG PHÁP KHỬ GAUSS")
    print("="*50)
    print_matrix(Ab, "Ma trận mở rộng [A|B] ban đầu", val_fmt)
    
    # 1. Quá trình thuận (Forward Elimination)
    for k in range(n):
        print(f"\n>>> BƯỚC LẶP k = {k} (Khử cột {k}) <<<")
        
        # Chọn phần tử chốt
        max_row = k
        for i in range(k + 1, n):
            if abs(Ab[i][k]) > abs(Ab[max_row][k]):
                max_row = i
                
        # Hoán vị dòng nếu cần
        if max_row != k:
            Ab[k], Ab[max_row] = Ab[max_row], Ab[k]
            print(f"[*] Đổi chỗ dòng {k} và dòng {max_row} (Chốt mới: {Ab[k][k]:{val_fmt}})")
        else:
            print(f"[*] Không cần đổi dòng (Chốt hiện tại: {Ab[k][k]:{val_fmt}})")
            
        if abs(Ab[k][k]) < 1e-12:
            raise ValueError("Ma trận hệ số suy biến!")
            
        # Khử các dòng phía DƯỚI dòng k
        for i in range(k + 1, n):
            factor = Ab[i][k] / Ab[k][k]
            print(f"    -> Khử dòng {i}: D_{i} = D_{i} - ({factor:{val_fmt}}) * D_{k}")
            for j in range(k, n + 1):
                Ab[i][j] -= factor * Ab[k][j]
                
        print_matrix(Ab, f"Ma trận sau khi khử xong cột {k}", val_fmt)
                
    # 2. Quá trình thế ngược (Backward Substitution)
    print("\n>>> QUÁ TRÌNH THẾ NGƯỢC <<<")
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(Ab[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (Ab[i][n] - sum_ax) / Ab[i][i]
        print(f"[*] Tìm được x[{i}] = ({Ab[i][n]:{val_fmt}} - {sum_ax:{val_fmt}}) / {Ab[i][i]:{val_fmt}} = {x[i]:{val_fmt}}")
        
    print("\n=> KẾT LUẬN NGHIỆM: [" + ", ".join([f"{val:{val_fmt}}" for val in x]) + "]")
    return x


def gauss_jordan_solve_verbose(A, B, val_fmt=".4f"):
    """
    Giải hệ Ax = B bằng phương pháp Gauss-Jordan.
    CÓ IN RA CÁC BƯỚC TRUNG GIAN.
    """
    n = len(A)
    Ab = [row[:] + [B[i]] for i, row in enumerate(A)]
    
    print("\n" + "="*50)
    print(" BẮT ĐẦU PHƯƠNG PHÁP GAUSS-JORDAN")
    print("="*50)
    print_matrix(Ab, "Ma trận mở rộng [A|B] ban đầu", val_fmt)
    
    # Quá trình Gauss-Jordan
    for k in range(n):
        print(f"\n>>> BƯỚC LẶP k = {k} (Xử lý cột {k}) <<<")
        
        # Chọn phần tử chốt
        max_row = k
        for i in range(k + 1, n):
            if abs(Ab[i][k]) > abs(Ab[max_row][k]):
                max_row = i
                
        if max_row != k:
            Ab[k], Ab[max_row] = Ab[max_row], Ab[k]
            print(f"[*] Đổi chỗ dòng {k} và dòng {max_row} (Chốt mới: {Ab[k][k]:{val_fmt}})")
        else:
            print(f"[*] Không cần đổi dòng (Chốt hiện tại: {Ab[k][k]:{val_fmt}})")
        
        pivot = Ab[k][k]
        if abs(pivot) < 1e-12:
            raise ValueError("Ma trận hệ số suy biến!")
            
        # Chuẩn hóa dòng k
        print(f"[*] Chuẩn hóa dòng {k}: Chia toàn bộ dòng cho {pivot:{val_fmt}}")
        for j in range(k, n + 1):
            Ab[k][j] /= pivot
            
        # Khử tất cả các dòng KHÁC dòng k
        for i in range(n):
            if i != k:
                factor = Ab[i][k]
                if abs(factor) > 1e-12:  # Chỉ in nếu thực sự có phép trừ
                    print(f"    -> Khử dòng {i}: D_{i} = D_{i} - ({factor:{val_fmt}}) * D_{k}")
                    for j in range(k, n + 1):
                        Ab[i][j] -= factor * Ab[k][j]
                        
        print_matrix(Ab, f"Ma trận sau khi đưa chốt về 1 và khử cột {k}", val_fmt)
                    
    x = [Ab[i][n] for i in range(n)]
    print("\n=> KẾT LUẬN NGHIỆM (Nằm ở cột cuối cùng): [" + ", ".join([f"{val:{val_fmt}}" for val in x]) + "]")
    return x


# ==========================================
# KIỂM TRA CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    # Sử dụng hệ phương trình:
    #  2x +  y -  z =  8
    # -3x -  y + 2z = -11
    # -2x +  y + 2z = -3
    
    A_matrix = [
        [ 2.0,  1.0, -1.0],
        [-3.0, -1.0,  2.0],
        [-2.0,  1.0,  2.0]
    ]
    B_vector = [8.0, -11.0, -3.0]
    
    # Chọn format:
    # ">8.5f" nghĩa là: Căn lề phải (>), chiếm 8 khoảng trống, lấy 5 chữ số sau dấu phẩy (f)
    my_format = ">8.5f"
    
    # 1. Chạy Gauss
    try:
        gauss_solve_verbose(A_matrix, B_vector, val_fmt=my_format)
    except ValueError as e:
        print(e)
        
    # 2. Chạy Gauss-Jordan
    try:
        gauss_jordan_solve_verbose(A_matrix, B_vector, val_fmt=my_format)
    except ValueError as e:
        print(e)