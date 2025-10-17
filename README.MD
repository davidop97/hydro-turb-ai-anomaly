### Pasos de ejecución del proyecto

Ubicarse en la ruta del proyecto 

 C:\Users\Usuario\Desktop\tests

Creación del ambiente virtual con el comando:

python -m venv venv

Activar el entorno virtual

.\venv\Scripts\activate

### Uso de librería pipeline

Dirgirse a la ruta de la librería 

C:\Users\Usuario\Desktop\turbine_pipeline

Con el entorno virtual activado ejecutar 

pip install .

Con el comando anterior se instalará en modo no editable

Si se quiere hacer uso en modo editable usar 

pip install -e .


Configurar la raiz del proyecto
set PYTHONPATH=C:\Users\Usuario\Desktop\tests;%PYTHONPATH%


