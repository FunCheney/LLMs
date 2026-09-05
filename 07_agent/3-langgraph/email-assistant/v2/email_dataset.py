EVAL_DATASET = [
    {
        "inputs": {
            "email_content": """
Hi,

I forgot my password and cannot log into my account.

How can I reset it?

Thanks!
"""
        },
        "outputs": {
            "intent": "question"
        },
    },
    {
        "inputs": {
            "email_content": """
Hi,

I found a serious bug in the application.
The page crashes whenever I open my profile.

Thanks.
"""
        },
        "outputs": {
            "intent": "bug"
        },
    },
]
