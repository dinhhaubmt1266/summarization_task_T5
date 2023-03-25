from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments

tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")  
model = AutoModelForSeq2SeqLM.from_pretrained("C:/dclv/nlp_extend/django_web/web_demo/model_Transformer5_SummaryText")

# model.cuda()

sentence = """
Lực lượng đoàn viên, thanh niên trong độ tuổi lao động chiếm tỷ lệ khá lớn, nhìn chung lực lượng này trình độ tay nghề của thanh niên còn thấp. Để có nguồn thu nhập, thanh niên xuất khẩu lao động sang các nước như Hàn Quốc, Nhật Bản… Tuy nhiên, một số công ty thu hút lao động hứa hẹn sau khi lao động có thu nhập cao nhưng không đúng như trong hợp đồng, đã có nhiều trường hợp đoàn viên, thanh niên tay trắng trở về và gánh nặng những khoản nợ đã vay để làm các khoản chi phí cho việc xuất khẩu lao động. Đề nghị lãnh đạo các cấp có biện pháp định hướng và cung cấp thông tin từ các doanh nghiệp nhà nước có đủ năng lực đưa lao động đi làm việc tại nước ngoài hợp pháp."""


text =  "vietnews: " + sentence + " </s>"
encoding = tokenizer(text, return_tensors="pt")
# input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=256,
    early_stopping=True
)
for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(line)