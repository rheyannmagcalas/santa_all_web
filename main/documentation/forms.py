# from django import forms
# from documentation.models import UserProfileInfo
# from django.contrib.auth.models import User
# from django.contrib.auth.hashers import check_password
# from django.utils import timezone

# class UserForm(forms.ModelForm):
#     password = forms.CharField(widget=forms.PasswordInput())
    
#     class Meta():
#         model = User
#         fields = ('username','password','email')

#     def clean(self):
#         cleaned_data = super(UserForm, self).clean()
#         confirm_password = cleaned_data.get('password')
#         if not check_password(confirm_password, self.instance.password):
#             self.add_error('confirm_password', 'Password does not match.')
    
#     def save(self, commit=True):
#         user = super(UserForm, self).save(commit)
#         user.last_login = timezone.now()
#         if commit:
#             user.save()
#         return user

# class UserProfileInfoForm(forms.ModelForm):
#     birth_date = forms.DateField(widget=forms.DateInput(format = '%d/%m/%Y'), 
#                                  input_formats=('%d/%m/%Y',))
#     class Meta():
#         model = UserProfileInfo
#         fields = ('birth_date',)