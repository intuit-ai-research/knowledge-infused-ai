question_prompt = """
You are a question generator. Given a series of pargraphs, your job is to generate questions for each paragraph.

The question generated should be able to be answered ONLY based on the information in the paragraph. 
The question generated should be about the main topic of the paragraph.

##Paragraphs:
{passed_in_data}

Return the questions in a list.
["1. question 1", "2. question 2", "3. question 3" ...]

##Questions:
"""
