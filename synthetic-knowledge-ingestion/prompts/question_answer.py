question_answer_prompt = """
You are a question generator. Given a series of pargraphs, your job is to generate questions for each paragraph and construct the corresponding answers.

The question generated should be able to be answered ONLY based on the information in the paragraph. 
The question generated should be about the main topic of the paragraph.
The answer should be about the generated question and based on the information in the paragraph.

##Paragraphs:
{passed_in_data}

Return the questions in a list of json.
[{"q": "question 1", "a": "answer 1"}, {"q": "question 2", "a": "answer 2"}, ...]

##Questions and Answers:
"""
