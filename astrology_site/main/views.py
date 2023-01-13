import os

from django.shortcuts import render

from .adapter import AdapterML

sign_list = ['Водолей', 'Рыбы', 'Овен', 'Телец', 'Близнецы', 'Рак', 'Лев', 'Дева', 'Весы', 'Скорпион', 'Стрелец', 'Козерог']


def parse_ml_output(text: str, sign: str) -> str:
    text = text.replace(f'{sign}', f'{sign}: ')
    text = text.replace('[SG]', '')
    text = text.replace('[EG]', '')
    text = text.split('"')
    return ''.join(text[:-1])


def check_input(text: str) -> bool:
    if text.isalpha():
        return True
        # details = detect(text)
        # if details == 'ru' or details == 'bg':
        #     return True
    return False


def index(request):
    result = ''
    if request.method == 'POST':
        sign = f"[SG]{list(request.POST)[1]} "
        if not os.path.exists(f"./checkpoint-24500"):
            return render(request, 'main/main.html', {'result': 'Какое-то рандомное предсказание'})
        ml = AdapterML('checkpoint-24500')
        result = parse_ml_output(ml.generate(sign), sign)
        return render(request, 'main/main.html', {'result': result})

    return render(request, 'main/main.html', {'result': result})
