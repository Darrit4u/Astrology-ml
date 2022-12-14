import os

from django.shortcuts import render
from langdetect import detect

from .adapter import AdapterML


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
        input_sign = request.POST['sign']
        if check_input(input_sign):
            if not os.path.exists(f"./checkpoint-24500"):
                return render(request, 'main/main.html', {'result': 'Какое-то рандомное предсказание'})
            sign = f"[SG]{request.POST['sign']} "
            ml = AdapterML('checkpoint-24500')
            result = parse_ml_output(ml.generate(sign), sign)
            return render(request, 'main/main.html', {'result': result})
        error_mes = 'Пожалуйста, введите ваш знак зодиака на русском языке и буквам'
        return render(request, 'main/main.html', {'error_message': error_mes, 'result': result})

    return render(request, 'main/main.html', {'result': result})
