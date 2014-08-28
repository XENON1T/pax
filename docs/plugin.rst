========
Plugins
========

Intro
#####

This is a step by step guide to creating a plugin. All source code is located in plugin directories, which are found automatically (subdirectories are also searched automatically). You can also define custom plugin search paths using the `plugin_paths` variable to the configuration..

Creating the Class
##################

Plugins are always derived from one of the plugin base classes. These classes are InputPlugin, OutputPlugin, and TransformPlugin. Intput and output plugins are used to define different methods of data input and output (ex. input via file, MongoDB, etc.). Transform plugins are used to define intermediate processing steps on the data.

Every plugin has two required functions: ::

  startup(self)
  shutdown(self)


which are like a constructor and destructor, repectively. The constructor should be used to initialize any member variables while the destructor can be used to close them cleanly. The different plugin types also have different required functions in addition which are described in the following sections.

The configuration dictionary is available from self.config.

In order to run with a plugin it should be added to your copy of the bin/paxit script. In this script the plugins are defined as lists in the pax.processor function. You do this also via the configuration: ::

  [pax]
  input = 'class'
  transform=['class_name']
  my_postprocessing = ['class_name']
  output = ['class_name']


Your plugin should be added to the proper list. The lists are processed in order so if your plugin depends on data fields that are added to the event by other plugins please make sure your plugin is positioned after.  You do not need to specify each field, such as 'input', but then a default will be used.  In nearly all cases, you will want to redefine only my_postprocessing. ::

  [pax]
  my_postprocessing = [ 'MyNewClassThatDoesSomethingTransform' ]

Transform Plugins
-----------------

Most user-defined plugins are probably transform plugins. These are used to perform processing steps on the data. Every transform plugin should override the ::

  transform_event(self,event):

function. All modification steps should be included in this function and it must return the modified event object.

Input Plugins
--------------

Input plugins are used to define input sources, which can include files or databases. These plugins must override the ::
  
  get_events(self):

Function. This function should yield an event object. The constructor and destructor should be used to open and close the input object.

Output Plugins
--------------

Output plugins are for writing data to file, database, or other output format. These plugins must override the ::

  write_event(self,event):

function. The functionality for saving the event should be contained within. It is advisable to use the constructor to define the output source and the destructor to close it.

Plugin-specific options
#######################

Plugins should not include hard-coded constants or other numerical values. These should be included as user-configurable options in the initialization files. Plugin options should be clearly defined in pax/default.ini. An options must include the option name, a default value, and a description of what the option does as a comment (comments are defined with the '#' character). The user then has the option to overload these options in their own configuration file at run time.



