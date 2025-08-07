
def get_similiar_script(original_script: str, new_subject) -> str:
    return f"""
    this is a video script which describe a value  and have the main features defines this value and short intersting description what is it 
    {original_script}
    use this and make the exact idea to create transcript for {new_subject}
    """

def get_description_prompt() -> str:
    return """make description that looks like this :
    What if responsibility wasn’t a burden—but a superpower?


    In this powerful visual short, we journey through what it truly means to be responsible—not just keeping promises, but leading with integrity, owning your role, and rising above blame. From lifting metaphorical weights in the gym to planting the seeds of impact, each scene explores one essential pillar of true responsibility: Initiative, Discipline, Self-Control, Influence, and more.

    This is not just a value. It's a way of life for those who choose to lead from within.

    👉 Watch till the end—and ask yourself: What weight are YOU willing to lift?

    Tags 
    responsibility, personal growth, self discipline, mindset, motivation, character development, emotional intelligence, integrity, take initiative, stop blaming, inspirational short, values education, hero mindset, self mastery, accountability, life lessons, animated motivation, self help, leadership values, be the change

    but for the current video script
    """


def get_hook_prompt() -> str:
    return """[A short description of the symbolic moment, action, or setting] +
[in the tone/style you want: cinematic, surreal, soft, minimalist, dreamlike, detailed] +
[use emotional or sensory adjectives: glowing, slow, peaceful, golden light, still]
""" 