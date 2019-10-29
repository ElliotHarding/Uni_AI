# Create a Kernel object. No string encoding (all I/O is unicode)
import aiml
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# Welcome user
print("Welcome to the plant pathology chat bot. Please feel free to ask questions to",
      "diagnose & cure any plant pathology which may be affecting your plant.")


