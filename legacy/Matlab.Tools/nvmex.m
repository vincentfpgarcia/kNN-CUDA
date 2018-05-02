function varargout = nvmex(varargin)
%MEX   Compile MEX-function
%
%   Usage:
%       nvmex [options ...] file [files ...]
%
%   Description:
%       MEX compiles and links source files into a shared library called a 
%       MEX-file, executable from within MATLAB. The resulting file has a
%       platform-dependent extension. Use the MEXEXT command to return the
%       extension for this machine or for all supported platforms.
%       
%       MEX accepts any combinations of source files, object files, and
%       library files as arguments.
%
%       The command line options to MEX are defined in the "Command Line
%       Options" section below.
%      
%       MEX can also build executable files for stand-alone MATLAB engine
%       and MAT-file applications. See the "Engine/MAT Stand-alone
%       Application Details" below for more information.
%       
%       You can run MEX from the MATLAB Command Prompt, Windows Command
%       Prompt, or the UNIX shell. MEX is a script named mex.bat on Windows
%       and mex on UNIX, and is located in the directory specified by
%       [matlabroot '/bin'].
%     
%       The first file name given (less any file name extension) will be the
%       name of the resulting MEX-file. Additional source, object, or
%       library files can be given to satisfy external references. On UNIX,
%       both C and Fortran source files can be specified when building a
%       MEX-file. If C and Fortran are mixed, the first source file given
%       determines the entry point exported from the MEX-file (MATLAB loads
%       and runs a different entry point symbol for C or Fortran MEX-files).
%     
%       MEX uses an options file to specify variables and values that are
%       passed as arguments to the compiler, linker, and other tools (e.g.
%       the resource linker on Windows). Command line options to MEX may
%       supplement or override contents of the options file, or they may
%       control other aspects of MEX's behavior. For more information see
%       the "Options File Details" section below.
%             
%       The -setup option causes MEX to search for installed compilers and
%       allows you to choose an options file as the default for future
%       invocations of MEX.
%
%       For a list of compilers supported with this release, refer to
%       Technical Note 1601 at:
%       http://www.mathworks.com/support/tech-notes/1600/1601.html
%
%   Command Line Options Available on All Platforms:
%       -<arch>
%           Build an output file for architecture <arch>. To determine the
%           value for <arch>, type "computer('arch')" at the MATLAB Command
%           Prompt on the target machine. Note: Valid values for <arch>
%           depend on the architecture of the build platform.
%       -ada <sfcn.ads>
%           Use this option to compile a Simulink S-function written in Ada,
%           where <sfcn.ads> is the Package Specification for the
%           S-function. When this option is specified, only the -v (verbose)
%           and -g (debug) options are relevant. All other options are
%           ignored. See [matlabroot '/simulink/ada/examples/README'] for
%           examples and information on supported compilers and other
%           requirements.
%       -argcheck
%           Add argument checking. This adds code so that arguments passed
%           incorrectly to MATLAB API functions will cause assertion
%           failures. (C functions only.)
%       -c
%           Compile only. Creates an object file but not a MEX-file.
%       -compatibleArrayDims
%           Build a MEX-file using the MATLAB Version 7.2 array-handling API,
%           which limits arrays to 2^31-1 elements. This option is the
%           default. (See also the -largeArrayDims option.)
%       -D<name>
%           Define a symbol name to the C preprocessor. Equivalent to a
%           "#define <name>" directive in the source.
%       -D<name>=<value>
%           Define a symbol name and value to the C preprocessor. Equivalent
%           to a "#define <name> <value>" directive in the source.
%       -f <optionsfile>
%           Specify location and name of options file to use. Overrides
%           MEX's default options file search mechanism.
%       -g
%           Create a MEX-file containing additional symbolic information for
%           use in debugging. This option disables MEX's default behavior of
%           optimizing built object code (see the -O option).
%       -h[elp]
%           Print this message.
%       -I<pathname>
%           Add <pathname> to the list of directories to search for #include
%           files.
%       -inline
%           Inline matrix accessor functions (mx*). The MEX-function
%           generated may not be compatible with future versions of MATLAB.
%       -l<name>
%           Link with object library. On PC name will be expanded to
%           "<name>.lib" or "lib<name>.lib" and on UNIX to "lib<name>".
%       -L<directory>
%           Add <directory> to the list of directories to search for
%           libraries specified with the -l option.
%       -largeArrayDims
%           Build a MEX-file using the MATLAB large-array-handling API. This
%           API can handle arrays with more than 2^31-1 elements when
%           compiled on 64-bit platforms. (See also the -compatibleArrayDims
%           option.)
%       -n
%           No execute mode. Print out any commands that MEX would otherwise
%           have executed, but do not actually execute any of them.
%       -O
%           Optimize the object code. Optimization is enabled by default and
%           by including this option on the command line. If the -g option
%           appears without the -O option, optimization is disabled.
%       -outdir <dirname>
%           Place all output files in directory <dirname>.
%       -output <resultname>
%           Create MEX-file named <resultname>. The appropriate MEX-file
%           extension is automatically appended. Overrides MEX's default
%           MEX-file naming mechanism.
%       -setup
%           Interactively specify the compiler options file to use as the
%           default for future invocations of MEX by placing it in the user
%           profile directory (returned by the PREFDIR command). When this
%           option is specified, no other command line input is accepted.
%       -U<name>
%           Remove any initial definition of the C preprocessor symbol
%           <name>. (Inverse of the -D option.)
%       -v
%           Verbose mode. Print the values for important internal variables
%           after the options file is processed and all command line
%           arguments are considered. Prints each compile step and final link
%           step fully evaluated.
%       <name>=<value>
%           Supplement or override an options file variable for variable
%           <name>. This option is processed after the options file is
%           processed and all command line arguments are considered. See the
%           "Override Option Details" section below for more details.
%
%   Command Line Options Available Only on Windows Platforms:
%       @<rspfile>
%           Include contents of the text file <rspfile> as command line
%           arguments to MEX.
%
%   Command Line Options Available Only on UNIX Platforms:
%       -cxx
%           Specify that the gateway routine is in C, and use the C++ linker
%           to link the MEX-file. This option overrides the assumption that
%           the first source file in the list is the gateway routine.
%       -fortran
%           Specify that the gateway routine is in Fortran. This option
%           overrides the assumption that the first source file in the list
%           is the gateway routine.
%
%   Options File Details:
%       There are template options files for the compilers that are
%       supported by MEX.  These templates are located at
%       [matlabroot '\bin\win32\mexopts'] or
%       [matlabroot '\bin\win64\mexopts'] on Windows, or
%       [matlabroot '/bin'] on UNIX.
%       
%       These template options files are used by the -setup option to define
%       the selected default options file.
%
%   Override Option Details:
%       When using the command line option <name>=<value>, you may need to
%       use the shell's quoting syntax to protect characters such as spaces
%       that have a meaning in the shell syntax.  On Windows double quotes
%       are used (e.g., COMPFLAGS="opt1 opt2") and on UNIX single quotes are
%       used (e.g., CFLAGS='opt1 opt2').
%       
%       It is common to use this option to supplement variables already
%       defined. To do this refer to the variable by prepending a "$"
%       (e.g., COMPFLAGS="$COMPFLAGS opt2" on Windows or
%       CFLAGS='$CFLAGS opt2' on UNIX).
%
%   Engine/MAT Stand-alone Application Details:
%       For stand-alone engine and MAT-file applications, MEX does not use
%       the default options file; you must use the -f option to specify
%       an options file.
%       
%       The options files used to generate stand-alone MATLAB engine and
%       MAT-file executables are named *engmatopts.bat on Windows, or
%       engopts.sh and matopts.sh on UNIX, and are located in the same
%       directory as the template options files referred to in the "Options
%       File Details" section.
%
%   Examples:
%       The following command compiles "yprime.c", building a MEX-file:
%     
%           mex yprime.c
%     
%       When debugging, it is often useful to use "verbose" mode as well
%       as include symbolic debugging information:
%     
%           mex -v -g yprime.c
%
%   See also COMPUTER, DBMEX, LOADLIBRARY, MEXEXT, PCODE, PREFDIR, SYSTEM

% Copyright 1984-2006 The MathWorks, Inc.
% This is an autogenerated file.  Do not modify.

try
    [varargout{1:nargout}]=nvmex_helper(varargin{:});
catch
    err = rmfield(lasterror, 'stack');
    error(err);
end
