---
noteId: "8b2f00b0d9fe11f184076d7a82e62770"
tags: []

---

# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603855
- **Họ và tên:** Vương Quang Thắng
- **SEED** (in ở ô đầu notebook): 5743

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc_1.jpg`](hand_calc_1.jpg) [`hand_calc_2.jpg`](hand_calc_2.jpg) — 2 ảnh mặt trước sau của ảnh tính tay có các bước | Có |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.989   0.795      0.881        0.824   
2. BoW + MLP (deep learning)         0.990   0.875      0.929        0.889   
3a. Encoder (MiniLM) + kNN           0.846   0.929      0.885        0.878   
3b. Encoder (MiniLM) + LogReg        0.981   0.929      0.954        0.923   
3c. Encoder (MiniLM) + MLP head      0.963   0.929      0.945        0.895   
4a. LLM nhỏ zero-shot                0.850   0.850      0.850        0.850   
4b. LLM lớn zero-shot                0.810   0.850      0.829        0.829   
4c. LLM nhỏ few-shot                 0.842   0.800      0.821        0.821   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)           10.493       8457.209       0.058   
2. BoW + MLP (deep learning)        10.323      26607.173       0.351   
3a. Encoder (MiniLM) + kNN         105.284         19.880     131.142   
3b. Encoder (MiniLM) + LogReg      107.609         19.860      78.398   
3c. Encoder (MiniLM) + MLP head    107.603         38.185      30.214   
4a. LLM nhỏ zero-shot                0.000          1.786     421.514   
4b. LLM lớn zero-shot                0.000          2.787     342.988   
4c. LLM nhỏ few-shot                 0.000          1.691     411.966   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.152           0.0  
2. BoW + MLP (deep learning)          0.803           0.0  
3a. Encoder (MiniLM) + kNN          202.002           0.0  
3b. Encoder (MiniLM) + LogReg       178.123           0.0  
3c. Encoder (MiniLM) + MLP head      39.377           0.0  
4a. LLM nhỏ zero-shot               957.762           0.0  
4b. LLM lớn zero-shot               472.458           0.0  
4c. LLM nhỏ few-shot               1488.957           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế
Thí nghiệm 2.B — Bỏ ReLU (predict("dl_relu", ...)):   
Đoán: Mất tính phi tuyến thì mô hình suy biến về Logistic Regression, F1 sẽ giảm mạnh (~0.88)
Thực tế: F1 vẫn đạt mức cao 0.929 (ngang bằng với MLP có ReLU)  
Lệch vì: Dữ liệu SMS phân loại chủ yếu dựa trên các từ khóa độc lập tuyến tính; ngoài ra bộ tối ưu Adam giúp tìm nghiệm tốt hơn gradient descent thuần của Phần 1

Thí nghiệm 4.A — Model lớn (120b) vs. Model nhỏ (20b) (predict("llm_models", ...)):   
Đoán: Model 120B lớn hơn nên sẽ có F1 cao hơn hẳn model 20B   
Thực tế: Model 20B đạt 0.850, cao hơn model 120B (0.829)  
Lệch vì: Tập SUB quá nhỏ (40 tin), model 120B "suy nghĩ" quá sâu dẫn đến suy diễn sai 1 tin ham mang giọng điệu thân mật, làm tụt điểm F1   

Thí nghiệm 4.B — Few-shot prompting (predict("llm_fewshot", ...)):   
Đoán: Thêm 4 ví dụ mẫu sẽ định hướng tốt hơn, F1 sẽ tăng tiệm cận 1.0 
Thực tế: F1 giảm từ 0.850 (zero-shot) xuống 0.821  
Lệch vì: Đưa ví dụ gây ra hiện tượng thiên kiến mẫu (prompt bias), khiến mô hình phân loại nhầm các tin nhắn có cấu trúc khác với 4 mẫu được mồi 

## 4. Phân tích 50 case khó
BoW + LogReg tụt mạnh nhất (F1 từ 0.824 rơi xuống 0.286, accuracy còn 50%)  
Nhóm case khó nhất: Spam giọng tự nhiên (Natural spam): BoW đoán đúng 0% 
Né từ khoá (Obfuscation): BoW chỉ đạt 17% 
2 Case cụ thể & cơ chế sai:
Case #14 ("Hi, I'm working from home and making about £300 a day..." - spam): Toàn từ giao tiếp đời thường mang trọng số âm lớn kéo về ham, không có từ khóa spam kinh điển nên BoW đoán nhầm thành ham
Case #29 ("C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd..." - spam): Kỹ thuật leet-speak biến chữ thành token lạ nằm ngoài vocab 2.000 từ, vector $x$ toàn số 0 nên BoW đoán nhầm thành ham

## 5. Thử thách cá nhân

Mã SV: 2A202603855 - SEED = 5743.   
Đề bài: Tăng tối đa Recall cho mô hình phân loại trên dữ liệu lệch mà không làm F1 giảm quá 5%.   
Hạ ngưỡng phân loại xác suất predict_proba từ mặc định 0.5 xuống 0.28.   
Kết quả: Recall tăng vọt từ 0.795 lên 0.902, Precision giữ ở mức 0.925, F1 đạt mức cao 0.913.   
Giải thích: Dữ liệu có rất ít spam; hạ ngưỡng quyết định buộc mô hình nhạy cảm hơn, tránh bỏ lọt các tin nhắn rác độc hại.

## 6. Phương án cho sản phẩm thật
Chọn: 3c. Encoder (MiniLM) + MLP head.   Số liệu dẫn chứng:Độ trễ: p95} = 39.38ms .   Chất lượng: Đạt F1 = 0.945 (Recall = 0.929, Precision = 0.963).   
Chi phí: 0.0 USD/1k tin tiền API token  
Chấp nhận đánh đổi: Chậm hơn BoW nhưng đổi lại F1 cao hơn vượt trội và hiểu ngữ nghĩa để chống bẫy né từ; kém LLM một chút ở các ca lừa đảo dài nhưng bảo mật dữ liệu, không lo trễ mạng hay rủi ro Prompt Injection

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
