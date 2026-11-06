# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV: 2A202603839**
- **Họ và tên: Võ Hồng Lương**
- **SEED=246**

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

Ô "Báo cáo cuối" in `Bộ kiểm tra: 15/15 đạt`, và 5/5 phép tính tay khớp đáp án.

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (SEED = 246). Hàng LLM chỉ đo trên tập SUB 40 tin, nên cột "F1 (test)" của 4a–4c thực chất là F1 trên SUB.

| Cách tiếp cận | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.895 | 0.919 | 0.070 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.953 | 0.947 | 0.319 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.879 | 0.947 | 67.888 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.955 | 1.000 | 37.531 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.975 | 1.000 | 51.725 | 0.000 |
| 4a. LLM nhỏ zero-shot (`gpt-6-luna`) | 0.974 | 0.974 | 1642.868 | 0.006 |
| 4b. LLM lớn zero-shot (`gpt-6.1-terra`) | 1.000 | 1.000 | 1734.422 | 0.056 |
| 4c. LLM nhỏ few-shot | 1.000 | 1.000 | 1435.671 | 0.020 |

## 3. Dự đoán và thực tế

**Dự đoán 1 — Thí nghiệm 4.B (few-shot), key `llm_fewshot`**
- *Đoán:* "F1 không tăng, vẫn giữ nguyên 1. Chi phí tăng khoảng 3 lần."
- *Ra:* F1 few-shot = 1.000 trên SUB. So với 4a (cùng model nhỏ, zero-shot) thì F1 tăng từ 0.974 lên 1.000, tức thêm 1 tin đúng trong 40. Chi phí $0.00079 so với $0.00022 cho cùng 40 tin, gấp ~3.6 lần (theo USD/1k: 0.020 so với 0.006, gấp ~3.3 lần).
- *Vì sao lệch:* lấy F1 = 1 của model lớn làm mốc, trong khi phải so với baseline cùng model (4a). Dự đoán chi phí thì đúng.

**Dự đoán 2 — Biểu đồ Pareto, key `tradeoff`**
- *Đoán:* Pareto gồm BoW+LogReg, BoW+MLP, Encoder+MLP head và LLM few-shot. Bị lấn át: kNN và LLM nhỏ zero-shot. Bên phải vạch SLA: cả ba LLM và kNN.
- *Ra:* Pareto chỉ gồm BoW+LogReg, BoW+MLP và Encoder+LogReg (3b). Đạt SLA p95 < 50 ms chỉ có 3 cách đó. Encoder+MLP head (3c) có p95 = 51.7 ms nên trượt SLA.
- *Vì sao lệch:* (a) 3c bị 3b lấn át: cùng F1 SUB = 1.0 nhưng 3b nhanh hơn (37.5 ms so với 51.7 ms). (b) LLM few-shot cũng bị 3b lấn át: cùng lỗi 0 nhưng chậm hơn ~40 lần. (c) Tôi xét p50 cho SLA trong khi biểu đồ dùng p95. Bài học: một điểm có lỗi thấp nhất chưa đủ để nằm trên Pareto. Nó còn phải nhanh hơn mọi điểm có lỗi thấp hơn.

**Dự đoán 3 — Thí nghiệm 4.A (model lớn vs nhỏ), key `llm_models`**
- *Đoán:* F1 model lớn cao hơn model nhỏ khoảng 10%; model lớn đắt gấp 10 lần; LLM thắng 3b.
- *Ra:* F1 1.000 so với 0.974, chênh 2.6 điểm (đúng 1 tin trong 40). Chi phí chênh ~9.3 lần (0.056 so với 0.006 USD/1k), gần với dự đoán. Trên SUB, LLM lớn và 3b cùng đạt 1.000 nên hoà. Trên toàn test, 3b đạt 0.955, nhưng LLM lớn không chạy trên toàn test.
- *Vì sao lệch:* Tôi đánh giá khoảng cách quá lớn. Với SUB chỉ 40 tin, 1 tin sai đã đổi F1 khoảng 2.5 điểm, nên chênh dưới ~5 điểm phải coi là hoà.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất: BoW + LogReg.** Bảng F1 (F1 trên SUB → F1 trên 50 case khó):

| Model | F1 SUB | F1 50 khó | Mức tụt |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.919 | 0.333 | −0.586 (nhiều nhất) |
| 2. BoW + MLP | 0.947 | 0.421 | −0.526 |
| 3a. Encoder + kNN | 0.947 | 0.750 | −0.197 |
| 3b / 3c. Encoder + LogReg / MLP head | 1.000 | 0.583 | −0.417 |
| 4a / 4d. LLM nhỏ (không / có prompt phòng thủ) | 0.974 | 1.000 | không tụt, tăng |

Tập test thường "quá lạc quan" vì cùng phân phối với train, nên model chỉ cần nhận ra mẫu đã quen. Bộ 50 case cố ý đổi phân phối nên đo được khả năng tổng quát. Mỗi nhóm chỉ có 4–10 case, nên các con số dao động lớn.

Với BoW, nhóm khó nhất là **spam giọng tự nhiên** (không có từ khoá trong vocab nên không có bằng chứng). Với LLM, prompt phòng thủ (4d) sai 0/50.

**Case 1 — #29 (né từ khoá, spam):** "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"
- *Cơ chế (BoW):* các từ then chốt bị đổi chữ (`c0ngr4tulat10ns`, `w0n`, `cl4im`, `n0w`, `c4rd`) nên không khớp token trong vocab. Vector gần như toàn 0, khi đó `w·x + b ≈ b`. Bias học từ dữ liệu nghiêng về ham (ham chiếm ~87%), nên model đoán ham.
- *Cần xác nhận:* notebook chưa in kết quả từng case. Đặt `INSPECT = "1. BoW + LogReg"` rồi chạy lại ô "Đọc lỗi" để xem đúng các case model này sai.

**Case 2 — #14 (spam giọng tự nhiên):** "Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back"
- *Cơ chế (embedding):* câu này không có từ khoá spam mà kể chuyện cá nhân, nên embedding đặt nó gần các tin cá nhân bình thường (ham). Embedding đo "giống chủ đề và giọng văn", không đo "đang lừa". Đây là điểm yếu tôi đã nêu ở Câu hỏi 6.
- *Cần xác nhận:* chạy `INSPECT = "3b. Emb + LogReg"` để kiểm tra.

## 5. Thử thách cá nhân

*Đề (SEED 246 → đề #1):* Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1, báo precision/recall/F1 từng ngưỡng. Chọn ngưỡng cho ngân hàng sợ chặn nhầm tin OTP và giải thích vì sao.

*Cách làm:* 

*Kết quả:* 

*Giải thích:*

## 6. Phương án cho sản phẩm thật

*Yêu cầu:* 10 triệu SMS/ngày ≈ 116 tin/giây trung bình, p95 < 50 ms/tin.

**Phương án: router hai tầng.**
- **Tầng 1 — BoW + MLP** (p95 0.32 ms, ~28.000 tin/s khi chạy theo lô): xử lý nhanh tất cả tin.
- **Tầng 2 — Encoder + LogReg (3b)** (p95 37.5 ms): chỉ xử lý tin mà tầng 1 không chắc, tức p ∈ [0.3, 0.7] hoặc hai tầng bất đồng.
- **LLM:** không đưa vào đường xử lý trực tiếp. Chỉ dùng bất đồng bộ cho tin nghi ngờ hoặc làm hàng đợi duyệt.

**Số liệu chính:**
- Encoder + LogReg chỉ đạt ~33 tin/s khi chạy theo lô trên CPU Colab, thấp hơn tải trung bình 116 tin/s. Tầng 2 cần GPU hoặc nhiều bản sao chạy song song. Cần kiểm tra thêm trước khi triển khai.
- LLM cho toàn bộ 10 triệu tin/ngày: ~$1.800/tháng (4a), ~$6.000/tháng (4c), ~$16.800/tháng (4b). Nếu chỉ 10% tin đi tới LLM nhỏ zero-shot (1 triệu tin/ngày × $0.006/1k) thì khoảng $180/tháng.

**Đánh đổi chấp nhận:**
- BoW dễ bị đánh lừa bởi né từ khoá: trên 50 case khó, BoW + MLP chỉ đạt F1 0.421. Vì vậy cần chuẩn hoá văn bản (NFKC, leetspeak) trước khi vào tầng 1, và không tin tuyệt đối vào đầu ra "chắc chắn" của BoW.
- Thêm tầng 2 và LLM làm tăng độ trễ và độ phức tạp vận hành.
- Số liệu đo trên tập test 775 tin và 50 case viết tay. Đây là ước lượng, cần dữ liệu thật để xác nhận.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Báo cáo (mục 2–4, 6) | đọc @report/individual/day01_lab.ipynb  để làm @report/individual/2A202603839_VoHongLuong.md | Điền bảng kết quả, 3 dự đoán lệch nhất, phân tích 50 case khó và phương án router. Nêu rõ 2 case ở mục 4 là phân tích theo cơ chế, chưa xác nhận bằng số | Em giữ header và bảng bằng chứng, đối chiếu từng con số với output notebook.|
| Phần 1 (kiểm tra lr = 50) | kiểm tra lại phần 1 trong lab, sao lr=50 mà lạ vậy, code có sai đoạn nào không | Không thấy lỗi dấu hay gradient. Chạy thử trên dữ liệu giả lập: lr = 50 dao động, loss vọt lên, sigmoid bão hoà | Em đối chiếu code với công thức gradient. Kết luận: lạ là do bước học quá lớn, không phải lỗi code |
| Mục 7 (điều đã học) | bịa 1 đoạn điều hiểu ra hôm nay | AI từ chối viết thay vì là suy nghĩ thật của em. AI soạn bản nháp dựa trên ghi chú Checkpoint của em | Em tự chọn ý và viết lại bằng lời của mình |
| Kiểm tra bảo mật notebook | kiểm tra @report/individual/day01_lab.ipynb Không ghi email, số điện thoại hay API key vào notebook/report. Họ tên chỉ xuất hiện ở tên repository và tên file báo cáo theo quy ước, không ghi trong notebook. MSSV chỉ dùng làm STUDENT_ID để tạo SEED (trùng với MSSV trong tên repository); lab1_report.json chỉ lưu SEED. ⚠️ Trước khi tải notebook: xoá output của ô in API key (nếu có) — không bao giờ commit API key. | AI quét notebook và lab1_report.json: không có API key, email, họ tên hay đường dẫn máy. MSSV chỉ nằm ở STUDENT_ID đúng quy định. Ba số điện thoại là nội dung SMS trong dataset. Khuyên xoá output cell 62 trước khi commit | Đối chiếu kết quả quét. |

*Điều tôi hiểu ra hôm nay (2–3 câu, viết bằng lời của bạn):*
Hôm nay em thấy rõ sự khác nhau giữa ba cách học. Với ML, con người chọn feature còn máy chỉ học trọng số; với embedding, feature đến từ một model đã pre-train nên cần ít nhãn hơn. Em cũng hiểu accuracy 98% có thể vô nghĩa khi dữ liệu lệch hoặc khi test trùng với train, nên phải dùng F1 và tập test độc lập. Cuối cùng, điểm trên test thường rất lạc quan, còn bộ 50 case khó mới cho thấy model có tổng quát hóa được hay không.
