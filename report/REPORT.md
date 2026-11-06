# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU (Colab không cấp GPU cho phiên này — ô setup in `Device: cpu`, tức `torch.cuda.is_available() = False`)

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145 (bản ghim trong notebook, khớp trường `ultralytics_version` của cả ba JSON)

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

Checksum checkpoint do ô setup kiểm tra và ghi vào JSON (`model_sha256`):

| Checkpoint | sha256 (12 ký tự đầu) |
| --- | --- |
| `yolo11n-cls.pt` | `c62d41bf9625…` |
| `yolo11n.pt` | `0ebbc80d4a76…` |
| `yolo11n-seg.pt` | `55ed65c56c91…` |

Ảnh mẫu COCO val2017, checksum khớp: `traffic` = COCO 210273 (`3ec23de63592…`), `kitchen` = COCO 397133
(`09e1d25c75f7…`), `dining` = COCO 166918 (`a7f8457580a2…`).

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (giữ nguyên mặc định của notebook)

**Thay đổi so với notebook nguồn:** Không. Chạy **Run all** trên Google Colab theo đúng notebook phát
hành, không sửa code, không đổi checkpoint, không đổi threshold. Ô validation cuối báo `PASS`.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

**1.** Record hạng 1 (`sample_id = traffic`, `coco_image_id = 210273`, ảnh 640 × 428):
`class_id`=468 ; `class_name`=`cab` ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=`ImageNet-1K`.
Bốn dòng còn lại của top-5: `minibus` 0.164284, `police_van` 0.085848, `recreational_vehicle` 0.054110,
`streetcar` 0.048193.

**2.** Record này nói về **cả bức ảnh**. Trong JSON không có một trường nào định vị vật thể — không có
`bbox_xyxy`, không có `polygon_xy`, không có `instance_id`; chỉ có `image_width`/`image_height` cho toàn
ảnh và một lớp kèm `rank`. Nhìn `classification_top5.png` thì rõ hơn: ảnh có hàng chục xe buýt, ô tô, người
đi bộ, nhưng output vẫn chỉ là **một** danh sách 5 lớp xếp hạng cho cả khung hình, không chỉ ra “chiếc taxi
nào”. Đơn vị nhãn ở đây là **một ảnh = một nhãn**.

**3.** Class list đi kèm **taxonomy của checkpoint**, không phải model tự nghĩ ra. `yolo11n-cls.pt` được
huấn luyện trên ImageNet-1K nên nó chỉ có thể trả về tên nằm trong 1000 lớp đó; JSON ghi thẳng
`taxonomy_name = "ImageNet-1K"`. Vì ImageNet không có lớp kiểu “đường phố đông xe”, model buộc phải chọn
lớp gần nhất trong danh sách — nên top-5 toàn là loại xe (`cab`, `minibus`, `police_van`…) dù ý nghĩa thật
của ảnh là một cảnh giao thông. Giới hạn nằm ở taxonomy, không phải ở “model đoán dở”.

**4.** Vì tên lớp không đủ định danh. `class_id` là khóa thật mà model/dataset dùng, `class_name` chỉ là
nhãn người đọc, còn `taxonomy_name` cho biết cuốn từ điển nào đang được tra. Thiếu `taxonomy_name` thì
`cab` trong ImageNet-1K và `car` trong COCO-80 bị đem so với nhau dù khác định nghĩa; thiếu `class_id` thì
khi taxonomy đổi phiên bản (lớp bị gộp/đổi tên) không còn cách nào truy lại record cũ. Ghi cả ba là điều
kiện tối thiểu để nhãn có thể tái lập và đối chiếu giữa các bộ dữ liệu.

**5.** Guideline phải nói rõ **chọn lớp theo cái gì khi ảnh có nhiều chủ đề**, theo thứ tự quyết định:
(a) định nghĩa “chủ thể chính” — ví dụ vật chiếm diện tích lớn nhất, hoặc vật ở tiền cảnh/đúng tiêu cự;
(b) thứ tự ưu tiên khi vẫn hòa — ví dụ “phương tiện ưu tiên hơn hạ tầng đường”; (c) danh sách lớp hợp lệ
và các cặp lớp dễ nhầm kèm ví dụ ảnh (`cab` vs `minibus`); (d) khi không có lớp nào đúng hoặc hai chủ thể
ngang nhau thì gắn cờ escalation thay vì tự chọn bừa. Ảnh `traffic` là ví dụ: không có quy tắc “chủ thể
chính” thì hai annotator có thể gắn `cab` và `minibus` mà cả hai đều “có lý”.

**6.** `score = 0.510915` chỉ là mức tin của model trên phân phối dữ liệu huấn luyện của nó, không phải
bằng chứng đúng/sai. Nó được tính so với taxonomy ImageNet chứ không so với guideline của dự án; 0.51 cũng
có nghĩa gần một nửa xác suất rơi vào chỗ khác. Lấy score làm ground truth là nhận lại đúng bias của model
rồi dùng chính nó để huấn luyện/đánh giá model — vòng lặp tự xác nhận, và mọi lỗi hệ thống (như toàn bộ
cảnh giao thông bị quy về một loại xe) sẽ được đóng dấu “đúng”.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen` (COCO 397133, ảnh 640 × 427). Ngưỡng JSON = 0.35 → 11 prediction.

**1.** Record đã chọn: `class_name`=`person` ; `score`=0.912625 ;
`bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
(`taxonomy_name`=COCO-80, `coordinate_unit`=pixel, `bbox_format`=xyxy, `score_threshold`=0.35).

**2.** Gốc tọa độ `(0, 0)` ở góc trên-trái ảnh, x tăng sang phải, y tăng xuống dưới. Hộp bắt đầu ở góc
trên-trái `(385.33, 69.24)` — tức cách mép trái 385 px và cách mép trên 69 px, ngay trên đầu người đầu bếp
— và kết thúc ở góc dưới-phải `(498.92, 348.92)`, tại mũi giày. Hộp rộng 498.92 − 385.33 = 113.58 px và cao
348.92 − 69.24 = 279.68 px, tức chiếm khoảng 17.7 % chiều rộng và 65.5 % chiều cao ảnh.

**3.** Số prediction trên ảnh `kitchen` theo ngưỡng (ô so sánh trong notebook):

| Ngưỡng | Số prediction | Lớp |
| --- | ---: | --- |
| 0.20 | 17 | person ×2, bowl ×5, oven ×2, cup ×2, spoon ×3, potted plant, dining table, bottle |
| 0.35 | 11 | person ×2, bowl ×5, oven ×2, cup ×2 |
| 0.60 | 6 | person ×2, bowl ×2, oven ×2 |

Hạ từ 0.60 xuống 0.20 thì số hộp tăng gần ba lần (6 → 17) và xuất hiện thêm các lớp vật nhỏ/bị che
(`spoon`, `bottle`, `potted plant`). Cùng một ảnh, cùng một checkpoint — chỉ con số lọc thay đổi.

**4.** Khi hạ ngưỡng, reviewer phải xử lý phần đuôi điểm thấp: kiểm tra từng hộp mới xem có phải vật thật
không (6 hộp thêm ở mức 0.20 gồm cả `bottle` và `potted plant` rất đáng ngờ), loại hộp trùng/chồng lên cùng
một vật, sửa hộp bám sai biên, và quyết định lớp cho các vật nhỏ, mờ, bị che mà trước đó không xuất hiện.
Khối lượng xem tăng theo số hộp, còn tỷ lệ hộp sai trên tổng số cũng tăng — nên thời gian QC mỗi ảnh tăng
nhanh hơn số hộp. Đổi lại, ngưỡng cao thì reviewer phải tự phát hiện vật bị bỏ sót, việc này tốn công hơn
là xóa hộp thừa.

**5.** Quy tắc box (áp dụng cho nhiều ảnh): *“Hộp phải là hình chữ nhật nhỏ nhất bao trọn **phần nhìn thấy**
của vật thể, bám sát biên không quá 2 px đệm; phần bị che hoặc nằm ngoài khung ảnh thì không đoán thêm. Vật
bị khung ảnh cắt vẫn gắn nhãn, và cạnh hộp được đặt đúng tại mép ảnh (x = 0, y = 0, x = W, y = H). Mỗi vật
thể một hộp riêng, kể cả khi nhiều vật cùng lớp nằm sát nhau; không gộp một nhóm vật vào một hộp.”*

**6.** Trên `detection_predictions.png`, hộp `person` thứ hai ở mép trái —
`bbox_xyxy = [0.12, 263.18, 61.55, 310.78]`, `score = 0.61093` — chỉ chứa **bàn tay và cẳng tay** của một
người thứ hai; phần còn lại của người này nằm ngoài khung ảnh, nên `x_min = 0.12` nằm ngay sát mép trái.
Đây là **ca dùng guideline, không cần escalation**: tình huống “vật bị khung ảnh cắt, chỉ còn một bộ phận”
lặp lại ở hầu hết ảnh nên phải có quy tắc chung — chẳng hạn “gắn nhãn phần nhìn thấy nếu còn nhận ra được
loại vật; nếu phần nhìn thấy nhỏ hơn một ngưỡng cố định (ví dụ 5 % diện tích ảnh) thì đánh dấu thuộc tính
`truncated = true` thay vì bỏ qua”. Cái **đáng escalation** lại là chuyện khác trên cùng ảnh: hai hộp
`oven` (0.686791 và 0.6343) nằm chồng lên khu bếp lò xây liền tường — không rõ guideline của dự án coi “cả
cụm bếp lò” là một đối tượng hay mỗi cánh cửa lò là một đối tượng. Quyết định đó đổi cả cách đếm nhãn nên
phải hỏi Lab Coach rồi mới chốt thành rule, không tự quyết từng ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Ngưỡng 0.35 → 11 instance (`kitchen-001` … `kitchen-011`).

**1.** Record đã chọn: `instance_id`=`kitchen-001` ; `class_name`=`person` ; `score`=0.899318 ;
`polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy` = `[446.0, 70.0]`, `[445.0, 71.0]`, `[444.0, 71.0]`,
`[443.0, 72.0]`, `[442.0, 72.0]` ; `bbox_xyxy` = [385.45, 66.44, 498.02, 348.58].

**2.** Box chỉ cho biết “vật nằm trong khung chữ nhật này”, còn polygon cho biết **hình dạng thật của
vật**: 348 điểm của `kitchen-001` bám theo mái tóc xoăn, vai, hai dây yếm, tà tạp dề và khe giữa hai chân —
những chỗ mà hình chữ nhật 113 × 280 px đều tính là “thuộc về người”. Nói cách khác, box trả lời *ở đâu*,
polygon trả lời *đến đâu*: nó tách pixel của vật khỏi pixel nền, nên dùng được cho đo diện tích/đường bao,
cho cảnh vật thể nằm chéo hoặc lồng vào nhau, và cho các vật dài mảnh như hai cái muôi treo tường
(`kitchen-008`, `kitchen-011`) mà box gần như toàn là nền.

**3.** `instance_id` là mã của **một vật thể cụ thể trong output của bài lab này**, sinh theo công thức
`<sample_id>-<số thứ tự>` để mỗi dòng JSON có một khóa duy nhất và trỏ lại đúng một đa giác. Nó **không
phải class ID**: `kitchen-008` và `kitchen-011` cùng `class_name = spoon` (cùng `class_id`) nhưng là hai
cái muôi khác nhau treo cạnh nhau, mỗi cái một đa giác (46 và 37 điểm) — class ID nói *loại gì*,
`instance_id` nói *cái nào*. Nó cũng **không phải tracking ID**: tracking ID giữ nguyên qua nhiều khung hình
để nói “vẫn là vật đó đang di chuyển”, còn ở đây số thứ tự chỉ do thứ tự model trả về trong **một** ảnh;
chạy lại với ảnh khác hoặc ngưỡng khác thì cùng một vật có thể mang số khác.

**4.** Quy tắc vẽ biên mask: *“Đa giác bám theo biên nhìn thấy của **một** instance, đặt điểm trên ranh giới
vật–nền với sai số ≤ 2 px; vật bị che chia thành nhiều mảnh rời thì vẽ các mảnh đó cho cùng một instance,
không nối tắt qua phần bị che và không lấn sang vật đứng trước. Nơi hai vật cùng lớp chạm nhau phải có hai
đa giác riêng, đường biên chung không được chồng lấn. Chỗ biên mờ do nhòe/thiếu sáng thì cắt theo mép trong
của vùng mờ (không phình ra nền) và ghi chú thuộc tính `boundary_uncertain`.”*

**5.** Trên `segmentation_prediction.png`, mask `dining table` (`kitchen-005`, score 0.578417, 598 điểm,
`bbox_xyxy = [0.97, 234.74, 345.86, 423.87]`) **lấn sang vùng bếp lò** mà `kitchen-007` (`oven`, score
0.489657, `bbox_xyxy = [0.25, 188.95, 194.35, 294.62]`) cũng đang nhận: hai hộp chồng nhau ở dải
x ≈ 0–194, y ≈ 235–295, và trên hình thì mảng tím/xanh phủ liền từ mặt bàn gỗ sang mặt bếp đen, không có
đường cắt rõ. Chỗ này là **escalation**, không phải rule: cạnh bàn và thành bếp dính nhau trong bóng tối
nên ngay cả nhìn bằng mắt cũng không chắc biên nằm ở đâu, và câu hỏi thật sự là câu hỏi định nghĩa —
“mặt bếp xây liền có tính là một phần của bàn không, hay là đối tượng `oven` riêng?”. Annotator không nên tự
chọn; cần Lab Coach chốt định nghĩa rồi bổ sung ví dụ vào guideline cho mọi ảnh bếp sau này.
(Một ca thứ hai đáng ghi: `kitchen-004` được gán `potted plant` 0.632167 cho bó thảo mộc **khô treo trên
tường** ở góc trên-trái — sai lớp, và đa giác 100 điểm gần như là hình chữ nhật ôm bbox chứ không bám lá.)

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một ảnh = một `class_id` + `class_name` lấy từ taxonomy đã chốt (ở đây ImageNet-1K), kèm `taxonomy_name`; không có tọa độ | `traffic` có hàng chục xe nhưng chỉ được một nhãn; top-1 `cab` 0.510915 trong khi top-2 `minibus` 0.164284 — không có lớp nào mô tả “cảnh giao thông” | Đọc guideline “chủ thể chính”, chọn đúng một lớp hợp lệ; ảnh đa chủ đề hoặc không có lớp phù hợp thì gắn cờ escalation thay vì đoán | Nhãn có nằm trong taxonomy không; có áp dụng đúng quy tắc chủ thể chính không; các cặp dễ nhầm (`cab`/`minibus`) có nhất quán giữa các ảnh không; tỷ lệ ảnh bị gắn cờ |
| Phát hiện vật thể | Mỗi vật thể nhìn thấy = một hộp `bbox_xyxy` pixel + một lớp COCO-80 (+ cờ `truncated`/`occluded`) | Hộp `person` [0.12, 263.18, 61.55, 310.78] chỉ có bàn tay ở mép trái; hai hộp `oven` chồng lên cụm bếp xây liền; ở ngưỡng 0.20 xuất hiện thêm `bottle`, `potted plant` khó xác nhận | Vẽ hộp bao sát phần nhìn thấy, mỗi vật một hộp, đánh dấu vật bị cắt/bị che, không bỏ sót vật nhỏ chỉ vì model không thấy | Hộp có bám biên không, có hộp trùng/thiếu không, lớp có đúng không, cờ `truncated`/`occluded` có nhất quán không — và kiểm tra bằng mắt trên ảnh gốc, không dựa vào `score` |
| Instance segmentation | Mỗi instance = một đa giác `polygon_xy` pixel + lớp + `instance_id` riêng; hai vật cùng lớp vẫn là hai record | Mask `dining table` (`kitchen-005`) lấn sang vùng `oven` (`kitchen-007`); `kitchen-004` gán `potted plant` cho bó thảo mộc khô treo tường với đa giác gần như chữ nhật | Vẽ biên theo pixel thật của từng instance, tách hai vật chạm nhau, ghi chú chỗ biên mờ, không nối tắt qua phần bị che | Mask có lấn sang vật khác hoặc sang nền không; hai vật cùng lớp có bị gộp làm một không; `instance_id` có duy nhất không; số điểm quá ít so với hình dạng (dấu hiệu vẽ ẩu) |

## 5. An toàn dữ liệu

**1.** Quy tắc: chỉ đưa lên Colab/GitHub công khai những ảnh và artifact có nguồn công khai, có giấy phép
rõ ràng và đã ghi attribution (ở đây là ba ảnh COCO val2017 CC BY 2.0 kèm `IMAGE_ATTRIBUTION.md`); tuyệt
đối không upload ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ, và không ghi họ
tên/MSSV/email/số điện thoại vào báo cáo hoặc output — định danh chỉ nằm ở tên repository.

**2.** Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi: dừng ngay, **không chạy tiếp, không commit, không push**,
báo Lab Coach/GV phụ trách lab qua kênh hỗ trợ của lớp; nếu lỡ push rồi thì báo ngay để được hướng dẫn gỡ
và xóa lịch sử, đồng thời không chia sẻ lại file đó cho người khác.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
