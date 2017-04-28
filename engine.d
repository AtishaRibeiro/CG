engine.o: engine.cc easy_image.hh ini_configuration.hh l_parser.hh \
 vector.hh
	$(CC) $(CXXFLAGS) -c $< -o $@
