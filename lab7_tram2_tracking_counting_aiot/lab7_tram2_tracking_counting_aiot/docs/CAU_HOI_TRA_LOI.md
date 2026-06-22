# Câu hỏi phân tích và trả lời gợi ý

## 1. Vì sao counting cần tracking?

Detection từng frame không biết đối tượng ở frame sau có phải cùng một người hay không. Tracking cung cấp ID và lịch sử vị trí, nhờ đó hệ thống chỉ đếm khi cùng một ID thực sự đi qua vạch.

## 2. `max_distance` quá thấp gây gì?

Track khó ghép với detection mới khi đối tượng di chuyển nhanh, dẫn đến đổi ID và có thể đếm trùng.

## 3. `max_distance` quá cao gây gì?

Hai đối tượng gần nhau có thể bị ghép nhầm ID.

## 4. Vì sao cần `max_missed`?

Model có thể bỏ sót vài frame do che khuất hoặc mờ. Giữ track trong một số frame giúp ID không bị mất ngay.

## 5. Vì sao không chỉ đếm số bbox?

Số bbox là số đối tượng tại một thời điểm, không phải số lượt đi qua cửa. Một người đứng lâu sẽ xuất hiện ở nhiều frame nhưng chỉ được đếm một lượt khi cắt vạch.

## 6. Occupancy có luôn chính xác không?

Không. `count_in - count_out` chỉ là ước tính. Sai ID, bỏ sót, đi sát nhau hoặc đi ngoài vùng quan sát đều có thể làm lệch số liệu.

## 7. Model có nên mở cửa tự động ngay khi thấy người?

Không nên dựa duy nhất vào một detection. Cần rule bổ sung, vùng an toàn, thời gian xác nhận và cơ chế xử lý khi confidence thấp.
