# Installation
file(GLOB PYTHON_SCRIPTS "*.py")
install(FILES ${PYTHON_SCRIPTS}
		DESTINATION interfaces/python)