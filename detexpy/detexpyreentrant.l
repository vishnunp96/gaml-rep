%option never-interactive
%option reentrant

%top{
/*
 * Copyright (c) 1986-2007 Purdue University1
 * All rights reserved.
 * 
 * Developed by:  Daniel Trinkle
 *                Department of Computer Science, Purdue University
 *                http://www.cs.purdue.edu/
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal with the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * o Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimers.
 * 
 * o Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimers in the
 *   documentation and/or other materials provided with the distribution.
 * 
 * o Neither the names of Daniel Trinkle, Purdue University, nor the
 *   names of its contributors may be used to endorse or promote products
 *   derived from this Software without specific prior written
 *   permission.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */


/*
 * detex [-e environment-list] [-c] [-l] [-n] [-s] [-t] [-w] [file[.tex] ]
 *
 *	This program is used to remove TeX or LaTeX constructs from a text
 *	file.
 */

#include <Python.h>

#include <string.h>
#include <stdio.h>

//#include <iostream>

#include "detex.h"

#ifdef KPATHSEA

#include <c-auto.h>
#include <kpathsea/c-auto.h>
#include <kpathsea/config.h>
#include <kpathsea/c-memstr.h>
#include <kpathsea/c-pathmx.h>
#include <kpathsea/c-std.h>
#include <kpathsea/tex-file.h>

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#else /* KPATHSEA */

#ifdef HAVE_STRING_H
#include <string.h>
#else
#include <strings.h>
#define	strrchr rindex
#endif
#ifndef MAXPATHLEN
#include <sys/param.h>
#endif
#ifdef OS2
#include <stdlib.h>
#endif

#ifdef WIN32
#include <fcntl.h>
#include <io.h>
#endif

#endif /* KPATHSEA */
}

%{
#undef IGNORE


FILE *stream; // Stream for saving lex output to string
char *mainTexDirectory;
#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR '\\' 
#else 
#define PATH_SEPARATOR '/'
#endif 


#define	LaBEGIN		if (fLatex) BEGIN
#define	IGNORE		if (fSpace && !fWord) fprintf(stream, " ") //putchar(' ')
#define	SPACE		if (!fWord) fprintf(stream, " ") //putchar(' ')
#define	NEWLINE		if (!fWord) fprintf(stream, "\n") //putchar('\n')
#define	LATEX		fLatex=!fForcetex
#define KILLARGS(x)	cArgs=x; LaBEGIN LaMacro
#define STRIPARGS(x)	cArgs=x; LaBEGIN LaMacro2
#define	CITE(x)		if (fLatex && !fCite) KILLARGS(x)

#define NO_MALLOC_DECL

void AddInclude(char *sbFile);
void ErrorExit(const char *sb1);
void UsageExit(void);
void IncludeFile(char *sbFile, yyscan_t yyscanner);
void InputFile(char *sbFile, yyscan_t yyscanner);
void SetEnvIgnore(const char *sbEnvList);
#ifndef KPATHSEA
void SetInputPaths(void);
#endif
void Warning(const char *sb1, const char *sb2);
int BeginEnv(const char *sbEnv);
int EndEnv(const char *sbEnv);
int InList(char *sbFile);
int SeparateList(char *sbList, char *rgsbList[] ,char chSep, int csbMax);
FILE *TexOpen(char *sbFile);
char *SafeMalloc(int cch, const char *sbMessage);
#ifndef KPATHSEA
//char *getenv(char *env);
#ifndef NO_MALLOC_DECL
char	*malloc();
#endif
#ifdef OS2
void	yyless(int);
void	OS2UsageExit(void);
#endif
#endif /* KPATHSEA */

char	*rgsbEnvIgnore[MAXENVS];	/* list of environments ignored */
char	*rgsbIncList[MAXINCLIST];	/* list of includeonly files */
char	*rgsbInputPaths[MAXINPUTPATHS];	/* list of input paths in order */
char	sbCurrentIgnoredEnv[CCHMAXENV];	/* current environment being ignored */
char	*sbProgName;			/* name we were invoked with */
#ifndef NOFILE /* might be defined in <sys/param.h> */
#define NOFILE 256
#endif
FILE	*rgfp[NOFILE+1];		/* stack of input/include files */
int	cfp = 0;			/* count of files in stack */
int	cOpenBrace = 0;			/* count of `{' in <LaMacro> and <LaMacro2> */
int	cArgs = 0;			/* argument connt in <LaArgs> */
int	csbEnvIgnore;			/* count of environments ignored */
int	csbIncList = 0;			/* count of includeonly files */
int	csbInputPaths;			/* count of input paths */
int	fLatex = 0;			/* flag to indicated delatex */
int	fWord = 0;			/* flag for -w option */
int	fFollow = 1;			/* flag to follow input/include */
int	fCite = 0;			/* flag to echo \cite and \ref args */
int	fSpace = 0;			/* flag to replace \cs with space */
int	fForcetex = 0;			/* flag to inhibit latex mode */
int fShowPictures = 0;  /* flag to show picture names */

int currBracesLevel = 0;
int footnoteLevel = -100;

#ifdef FLEX_SCANNER
/* flex has contexts for buffers that need to be switched when file changes
 * otherwise output contains imported files in reverse order.  Weird, but
 * true.
 */
YY_BUFFER_STATE rgsb[NOFILE + 1]; /* flex context stack */
int             csb = 0;		 /* depth of flex context stack */
#endif /* FLEX_SCANNER */


/* 
 * Flags for math mode begin/end shorthand usage
 * Only seem to need them for the $-sign environments...?
 */
	/* int inLaFormulaMode = 0; */
	/* int inLaDisplayMode = 0; */
int inDisplayMode = 0;
int inMathMode = 0;

%}

S	[ \t\n]*
W	[a-zA-Z]+
N	[+-]?(([0-9]+(\.[0-9]+)?)|(\.[0-9]+))
U	pt|pc|in|bp|cm|mm|dd|cc|sp
HU	{U}|em
HD	{S}(({N}{S}{HU})|(\\{W})){S}
HG	{HD}(plus{HD})?(minus{HD})?
VU	{U}|ex
VD	{S}(({N}{S}{VU})|(\\{W})){S}
VG	{VD}(plus{VD})?(minus{VD})?
Z	\*?

%Start Define IncludeOnly Input Normal Control
%Start LaBegin LaEnd LaEnv LaInclude
%Start LaMacro LaOptArg LaMacro2 LaOptArg2 LaVerbatim 
%start LaBreak LaPicture

%%
<Normal>"%".*		/* ignore comments */	;

<Normal>"\\begin"{S}"{"{S}"document"{S}"}""\n"*	{LATEX; IGNORE;}

<Normal>"\\begin"     /* environment start */	{LaBEGIN LaBegin; IGNORE;}

<LaBegin>{S}"{"{S}"verbatim"{S}"}"		{	if (BeginEnv("verbatim"))
												BEGIN LaEnv;
											else
												BEGIN LaVerbatim;
											IGNORE;
										}

<LaVerbatim>"\\end"{S}"{"{S}"verbatim"{S}"}" /* verbatim mode */	{BEGIN Normal; IGNORE;}
<LaVerbatim>[^\\]+				fprintf (stream, "%s", yytext);//ECHO;
<LaVerbatim>.					fprintf (stream, "%s", yytext);//ECHO;

<LaBegin>{S}"{"{S}"minipage"{S}"}"		{ 	KILLARGS(1);
											if (BeginEnv("minpage"))
												BEGIN LaEnv;
											else
												BEGIN LaMacro; /* Normal; */
											IGNORE;
										}

<LaBegin>{S}"{"{S}"table"{S}"}"{S}"["{W}"]"		{
													if (BeginEnv("table"))
														BEGIN LaEnv;
													else
														BEGIN Normal;
													IGNORE;
												}

<LaBegin>{S}"{"{S}"figure"{S}"}"{S}"["{W}"]"	{
													if (BeginEnv("figure"))
														BEGIN LaEnv;
													else
														BEGIN Normal;
													IGNORE;
												}
						
<LaBegin>{W}					{	if (BeginEnv(yytext))
										BEGIN LaEnv;
									else
										BEGIN Normal;
									IGNORE;
								}
	/*<LaBegin>"\n"					NEWLINE;*/
<LaBegin>.					;



<Normal>"\\("		/* formula mode */	{	if (BeginEnv("math"))
												BEGIN LaEnv;
											IGNORE;
										}
<Normal>"\\)"					IGNORE;
<LaEnv>"\\)"		 /* end environment */	{	if (EndEnv("math"))
													BEGIN Normal;
												IGNORE;
											}

<Normal>"\\["		/* display mode */	{	if (BeginEnv("displaymath"))
												BEGIN LaEnv;
											IGNORE;
										}
<Normal>"\\]"					IGNORE;
<LaEnv>"\\]"		 /* end environment */	{	if (EndEnv("displaymath"))
													BEGIN Normal;
												IGNORE;
											}

<Normal>"$$"		/* display mode */	{
											if(!inDisplayMode) { /* Not currently in display mode */
												if (BeginEnv("displaymath"))
													BEGIN LaEnv;
												inDisplayMode = 1; /* Start display mode */
											}
											else { /* Currently in display mode */
												inDisplayMode = 0; /* End display mode */
											}
											IGNORE;
										}
<LaEnv>"$$"		 /* end environment */	{
											if(!inDisplayMode) /* Not currently in display mode */
												REJECT;
											if (EndEnv("displaymath")) {
												BEGIN Normal;
												inDisplayMode = 0; /* End display mode */
											}
											IGNORE; 
										}


<Normal>"\\$"					fprintf (stream, "%s", yytext);//printf("$"); /* Is this an unneccessary hack? */
<Normal>"$"		/* math mode */		{
										if(!inMathMode) { /* Not currently in math mode */
											if (BeginEnv("math"))
												BEGIN LaEnv;
											inMathMode = 1; /* Start math mode */
										}
										else { /* Currently in math mode */
											inMathMode = 0; /* End math mode */
										}
										IGNORE;
									}
<LaEnv>"\\$"					{ 
									if(inMathMode) { /* Currently in math mode */
						  				IGNORE;
									}
						  			else {
										REJECT;
									}
								}
<LaEnv>"$"		 /* end environment */	{
											if(!inMathMode) /* Not currently in math mode */
												REJECT;
											if (EndEnv("math")) {
												BEGIN Normal;
												inMathMode = 0; /* End math mode */
											}
											IGNORE;
										}




<LaEnv>"\\end"  /* absorb some environments */	{LaBEGIN LaEnd; IGNORE;}
<LaEnv>"\n"+					;/*NEWLINE;*/
<LaEnv>.					;

<LaEnd>{W}		 /* end environment */	{
											if (EndEnv(yytext))
												BEGIN Normal;
											IGNORE;
										}
<LaEnd>"}"					{BEGIN LaEnv; IGNORE;}
	/*<LaEnd>"\n"					NEWLINE;*/
<LaEnd>.					;

<Normal>"\\kern"{HD}					;
<Normal>"\\vskip"{VG}					;
<Normal>"\\vspace"{Z}{S}"{"{VG}"}"		;
<Normal>"\\hskip"{HG}					;
<Normal>"\\hspace"{Z}{S}"{"{HG}"}"		;
<Normal>"\\addvspace"{S}"{"{VG}"}"		;
<Normal>"{"{N}"pt}"						; /* hack to fix \begin{minipage}{300pt} */

<Normal>"\\newlength"				{ KILLARGS(1); }
<Normal>"\\setlength"				{ KILLARGS(2); }
<Normal>"\\addtolength"				{ KILLARGS(2); }
<Normal>"\\settowidth"				{ KILLARGS(2); }
<Normal>"\\settoheight"				{ KILLARGS(2); }
<Normal>"\\settodepth"				{ KILLARGS(2); }
<Normal>"\\newsavebox"				{ KILLARGS(1); }
<Normal>"\\sbox"					{ KILLARGS(1); }
<Normal>"\\savebox"					{ KILLARGS(2); }
<Normal>"\\usebox"					{ KILLARGS(1); }
<Normal>"\\raisebox"				{ STRIPARGS(2); }
<Normal>"\\parbox"					{ KILLARGS(1); }
<Normal>"\\scalebox"				{ STRIPARGS(2); }
<Normal>"\\resizebox"{Z}			{ KILLARGS(2); }
<Normal>"\\reflectbox"				;
<Normal>"\\rotatebox"				{ KILLARGS(1); }
<Normal>"\\includegraphics"[^{]*	{ LaBEGIN LaPicture; }

<LaPicture>"{"						;
<LaPicture>[^{}]+					{ if(fShowPictures) { fprintf (stream, "%s", yytext); } } //printf("<Picture %s>", yytext);
<LaPicture>"\}"{S}"\n"+				BEGIN Normal;
<LaPicture>"\}"						BEGIN Normal;

<Normal>"\\definecolor"				{ KILLARGS(3); }
<Normal>"\\color"					{ KILLARGS(1); }
<Normal>"\\textcolor"				{ KILLARGS(2); }
<Normal>"\\colorbox"				{ KILLARGS(2); }
<Normal>"\\fcolorbox"				{ KILLARGS(3); }
<Normal>"\\pagecolor"				{ KILLARGS(1); }
<Normal>"\\foilhead"				{ STRIPARGS(1); }
<Normal>"\\addfontfeature"			{ KILLARGS(1); }
<Normal>"\\thispagestyle"			{ KILLARGS(1); }
<Normal>"\\addcontentsline"			{ KILLARGS(3); }

<Normal>"\\part"{Z}					; /*NEWLINE;*/
<Normal>"\\chapter"{Z}				; /*NEWLINE;*/
<Normal>"\\section"{Z}				; /*NEWLINE;*/
<Normal>"\\subsection"{Z}			; /*NEWLINE;*/
<Normal>"\\subsubsection"{Z}		; /*NEWLINE;*/
<Normal>"\\paragraph"{Z}			; /*NEWLINE;*/
<Normal>"\\subparagraph"{Z}			; /*NEWLINE;*/

<Normal>"\\bibitem"	    /* ignore args  */	{KILLARGS(1); IGNORE;}
<Normal>"\\bibliography"    /* of these \cs */	{KILLARGS(1); IGNORE;}
<Normal>"\\bibstyle"				{KILLARGS(1); IGNORE;}
<Normal>" "?"\\cite"				{KILLARGS(1);} /* kill space before */
<Normal>"\\documentstyle"			{LATEX; KILLARGS(1); IGNORE;}
<Normal>"\\documentclass"			{LATEX; KILLARGS(1); IGNORE;}
<Normal>"\\usepackage"				{KILLARGS(1); IGNORE;}
<Normal>"\\end"						{KILLARGS(1); IGNORE;}
<Normal>"\\hypersetup"				{KILLARGS(1);}
<Normal>"\\index"					{KILLARGS(1);}
	/*<Normal>"\\footnote"				{KILLARGS(1); SPACE;}*/
<Normal>"\\label"					{KILLARGS(1); IGNORE;}
<Normal>"\\nameref"					{CITE(1); IGNORE;}
<Normal>"\\pageref"					{CITE(1); IGNORE;}
<Normal>"\\pagestyle"				{KILLARGS(1); IGNORE;}
<Normal>"\\ref"						{CITE(1); IGNORE;}
<Normal>"\\setcounter"				{KILLARGS(2); IGNORE;}
<Normal>"\\addtocounter"			{KILLARGS(2); IGNORE;}
<Normal>"\\newcounter"				{ KILLARGS(1); }
<Normal>"\\stepcounter"				{ KILLARGS(2); }

<Normal>"\\fontspec"				{KILLARGS(1);}

<Normal>"\\footnote"(\[([^\]])+\])?"{"		{
												fprintf(stream, "("); //putchar('(');
												footnoteLevel = currBracesLevel;
												++currBracesLevel;
											}
<Normal>"\\verb" /* ignore \verb<ch>...<ch> */	{
													if (fLatex) {
														char verbchar, c;
														verbchar = input(yyscanner);
														while ((c = input(yyscanner)) != verbchar) {
															/*if (c == '\n')
															NEWLINE;*/
															fprintf(stream, "%c", c); //putchar(c);
														}
													}
													IGNORE;
												}

<Normal>"\\newcommand"				{ LATEX; KILLARGS(2); }
<Normal>"\\renewcommand"			{ LATEX; KILLARGS(2); }
<Normal>"\\newenvironment"			{ LATEX; KILLARGS(3); }

<Normal>"\\def"		/* ignore def begin */	{BEGIN Define; IGNORE;}
<Define>"{"					BEGIN Normal;
<Define>"\n"				NEWLINE;
<Define>.					;




<Normal>"\\include"	/* process files */	{LaBEGIN LaInclude; IGNORE;}
<LaInclude>[^{ \t\n}]+			{
									IncludeFile(yytext, yyscanner);
							   		BEGIN Normal;
								}
<LaInclude>"\n"+				NEWLINE;
<LaInclude>.					;

<Normal>"\\includeonly"			{BEGIN IncludeOnly; IGNORE;}
<IncludeOnly>[^{ \t,\n}]+		AddInclude(yytext);
<IncludeOnly>"}"				{
									if (csbIncList == 0)
										rgsbIncList[csbIncList++] = '\0';
									BEGIN Normal;
								}
<IncludeOnly>"\n"+				NEWLINE;
<IncludeOnly>.					;

<Normal>"\\input"				{BEGIN Input; IGNORE;}
<Input>[^{ \t\n}]+				{
									InputFile(yytext, yyscanner);
									BEGIN Normal;
								}
<Input>"\n"+					NEWLINE;
<Input>.						;

	/* escaping commands */
<Normal>"\\slash"				fprintf(stream, "/"); //putchar('/');

<Normal>\\(aa|AA|ae|AE|oe|OE|ss)[ \t]*[ \t\n}] /* handle ligatures */  {(void)fprintf (stream, "%.2s", yytext+1);}  //{(void)printf("%.2s", yytext+1);}
<Normal>\\[OoijLl][ \t]*[ \t\n}]		{(void)fprintf (stream, "%.1s", yytext+1);}  //{(void)printf("%.1s", yytext+1);}
<Normal>"\\linebreak"(\[[0-4]\])?		{NEWLINE;}	/*BEGIN LaBreak;*/

<Normal>\\[a-zA-Z@]+	/* ignore other \cs */	{BEGIN Control; IGNORE;}
<Normal>"\\ "							SPACE;
<Normal>"\\\\"{Z}(\[[^\]]*\])?			NEWLINE;
<Normal>\\.								IGNORE;
<LaBreak>[0-4]?					{
									if (yytext==NULL || strlen(yytext)==0 || atoi(yytext)>=0)
										NEWLINE;
									BEGIN Normal;
								}

<Control>\\[a-zA-Z@]+				IGNORE;
<Control>[a-zA-Z@0-9]*[-'=`][^ \t\n{]*		IGNORE;
<Control>"\n"+					{BEGIN Normal; /*NEWLINE;*/}
<Control>[ \t]*[{]+				{++currBracesLevel;BEGIN Normal; IGNORE;}
<Control>[ \t]*					{BEGIN Normal; IGNORE;}
<Control>.						{yyless(0);BEGIN Normal;}

<Normal>[\\|]	/* special characters */	IGNORE;
<Normal>[!?]"`"				IGNORE;
<Normal>~					SPACE;
<Normal>-{2,3}				fprintf(stream, "-"); //putchar('-');
<Normal>``					fprintf(stream, "\""); //putchar('"');
<Normal>`					fprintf(stream, "\\"); //putchar('\'');
<Normal>''					fprintf(stream, "\""); //putchar('"');
<Normal>,,					fprintf(stream, "\""); //putchar('"');

	/* braces */
<Normal>"{"					{ ++currBracesLevel; }
<Normal>"}"					{
								--currBracesLevel;
								if (currBracesLevel == footnoteLevel) {
									fprintf(stream, ")"); //putchar(')');
									footnoteLevel = -100;
								}
								SPACE;
							}
<Normal>{W}[']*{W}*			{
								if (fWord)
									(void)fprintf (stream, "%s\n", yytext);  //(void)printf("%s\n", yytext);
								else
									fprintf (stream, "%s", yytext);//ECHO;
							}
<Normal>[0-9]+				if (!fWord) fprintf (stream, "%s", yytext);//ECHO;
<Normal>.					if (!fWord) fprintf (stream, "%s", yytext);//ECHO;
<Normal>"\n"				if (!fWord) NEWLINE;
<Normal>("\t")+				if (!fWord) fprintf(stream, "\t"); //putchar('\t');

<LaMacro>"\["				{ BEGIN LaOptArg; }
<LaMacro>"{"				{ cOpenBrace++; }
<LaMacro>"}" /*"\n"{0,1}*/	{
								cOpenBrace--; /// Why was the newline part there?
								if (cOpenBrace == 0) {
									if (--cArgs==0)
										BEGIN Normal;
								}
							}
<LaMacro>.					;
<LaOptArg>"\]"				BEGIN LaMacro;	
<LaOptArg>[^\]]*			;

<LaMacro2>"\["				{ BEGIN LaOptArg2; }
<LaMacro2>"{"				{
								if (cOpenBrace == 0) {
									if (--cArgs==0) {
										BEGIN Normal;
										cOpenBrace--;
									}
								}
								cOpenBrace++;
							}
<LaMacro2>"}"				{ cOpenBrace--; }
<LaMacro2>.					;
<LaOptArg2>"\]"				BEGIN LaMacro2;	
<LaOptArg2>.				;

.|\n						; // Default rule, matches unknown text
%%


/////////////////////////////////////////////////////////////// detex_run
// Function we are exposing to Python
// Usage: sourcefile, dict_of_arguments
static PyObject * detex_run(PyObject *self, PyObject *args, PyObject *keywds) {

	const char *sbEnvList = DEFAULTENV; // Set the environment list to the default list
	
	/* get base name and decide what we are doing, detex or delatex */
	sbProgName = "detex";
	// Have we been called as delatex?
	if (strcmp("delatex",sbProgName) == 0)
	    fLatex = 1;

#ifndef KPATHSEA
	/* set rgsbInputPaths for use with TexOpen() */
	SetInputPaths();
#endif

	////////////////
	/// Get "arguments" from Python call
	////////////////
    static char *kwlist[] = {"file", "cite", "env", "latex", "follow", "space", "tex", "word", NULL};

    char *sourcefile;
	//char *env = sbEnvList;
	PyObject* env = NULL; // Variable to hold Python list of environments to ignore
	int numenv;

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|pO!ppppp", kwlist, &sourcefile, &fCite, &PyList_Type, &env, &fLatex, &fFollow, &fSpace, &fForcetex, &fWord)) {
        return NULL;
	}

	if(env != NULL) {
		numenv = PyList_Size(env);

		// Buffer to store env list
		char *bpenv;
		size_t sizeenv;
		// Memory stream for env list
		FILE *envstream = open_memstream (&bpenv, &sizeenv);

		PyObject *strObj;
		for(int i=0; i<numenv; i++) {
			strObj = PyList_GetItem(env, i);
			fprintf(envstream, "%s,", PyUnicode_AsUTF8(strObj));
		}

		fflush (envstream);
		fclose (envstream);

		sbEnvList = bpenv;

		//free(envstream);
		//free(bpenv);
		//free(sizeenv);
	}

	//printf("Env list: %s\n",sbEnvList);

	SetEnvIgnore(sbEnvList);

	char *lastSlash = NULL;
	lastSlash = strrchr(sourcefile, PATH_SEPARATOR); // you need escape character
	unsigned int lastSlashIndex = lastSlash - sourcefile + 1;
	if(lastSlashIndex > 0 && lastSlashIndex < strlen(sourcefile)) {
		mainTexDirectory = strndup(sourcefile, lastSlashIndex);
	}
	else {
		mainTexDirectory = "";
	}
	//printf("sourcefile: %s\n", sourcefile);
	//printf("lastSlashIndex: %d\n", lastSlashIndex);
	//printf("mainTexDirectory: %s\n", mainTexDirectory);


	// Buffer to store output to return
	char *bp;
	size_t size;
	// Memory stream to pass output to for return string
	stream = open_memstream (&bp, &size);



	yyscan_t yyscanner;
	struct yyguts_t * yyg;

	yylex_init(&yyscanner);
	yyg = (struct yyguts_t*)yyscanner;




	/* process input file */
    if ((yyin = TexOpen(sourcefile)) == NULL) {
		Warning("can't open file", sourcefile);
		return PyBytes_FromString(""); // Return empty string, as we could not open source
    }
	else {
		//yyout = open_memstream(&ptr, &size);

		//yyFlexLexer( yyin, ostream* open_memstream(&ptr, &size));

		//YY_FLUSH_BUFFER; // Clean buffer? Will this fix "input buffer overflow, can't enlarge buffer because scanner uses REJECT" error?
		//yyrestart(yyin);



		BEGIN Normal;
		//(void)yylex();



		yylex(yyscanner);

		yylex_destroy(yyscanner);
	}


#ifndef FLEX_SCANNER
	// Report unterminated document
	if (YYSTATE != Normal)
	    ErrorExit("input contains an unterminated mode or environment");
#endif

	fflush (stream);
	fclose (stream);

	//Py_INCREF(Py_None);
	//return Py_None;
	//printf(bp);

	//return PyUnicode_FromString(bp);
	return PyBytes_FromString(bp);

	//return PyUnicode_FromFormat("%s", bp);//PyUnicode_FromString(bp);
	//return PyUnicode_DecodeFSDefault(bp);
	//return PyUnicode_DecodeASCII(bp, PyUnicode_GET_LENGTH(PyUnicode_FromString(bp)), "strict");
}
///////////////////////////////////////////// End detex_run

#ifdef FLEX_SCANNER
#undef yywrap
#endif

/******
** yywrap -- handles EOF for lex.  Check to see if the stack of open files
**	has anything on it.  If it does, set yyin to the to value.  If not
**	return the termination signal for lex.
******/

int
yywrap(yyscan_t yyscanner)
{
	struct yyguts_t * yyg;
	yyg = (struct yyguts_t*)yyscanner;

	(void)fclose(yyin);
#ifdef FLEX_SCANNER
        /* Pop context state */
	if (csb > 0) {
		yy_delete_buffer( YY_CURRENT_BUFFER, yyscanner );
		yy_switch_to_buffer( rgsb[--csb], yyscanner );
	}
#endif /* FLEX_SCANNER */
	if (cfp > 0) {
	    yyin = rgfp[--cfp];
	    return(0);
	}
	return(1);
}

#ifdef OS2

/******
** yyless -- return characters to the input stream.  Some systems don't have
**	a yyless routine
******/

void
yyless(int n)
{
	int 	i = strlen(yytext);

	while (i > n) unput(yytext[--i]);
	yytext[yyleng = n] = '\0';
}
#endif

/******
** SetEnvIgnore -- sets rgsbEnvIgnore to the values indicated by the
**	sbEnvList.
******/

void
SetEnvIgnore(const char *sbEnvList)
{
	char *sb;
	sb = SafeMalloc(strlen(sbEnvList) + 1, "malloc for SetEnvIgnore failed");
	(void) strcpy(sb, sbEnvList);

	
	csbEnvIgnore = SeparateList(sb, rgsbEnvIgnore, CHENVSEP, MAXENVS);
	if (csbEnvIgnore == my_ERROR)
	    ErrorExit("The environment list contains too many environments");
}

/******
** BeginEnv -- checks to see if sbEnv is in the list rgsbEnvIgnore.  If it
**	is, sbCurrentIgnoredEnv is set to sbEnv.
******/

int
BeginEnv(const char *sbEnv)
{
	int	i;

	if (!fLatex) return(0);
	for (i = 0; i < csbEnvIgnore; ++i)
	    if (strcmp(sbEnv, rgsbEnvIgnore[i]) == 0) {
			(void)strcpy(sbCurrentIgnoredEnv, sbEnv);
			return(1);
	    }
	return(0);
}

/******
** EndEnv -- checks to see if sbEnv is the current environment being ignored.
******/

int
EndEnv(const char *sbEnv)
{
	if (!fLatex) return(0);
	if (strcmp(sbEnv, sbCurrentIgnoredEnv) == 0)
	    return(1);
	return(0);
}

/******
** InputFile -- push the current yyin and open sbFile.  If the open fails,
**	the sbFile is ignored.
******/

void
InputFile(char *sbFile, yyscan_t yyscanner)
{
	struct yyguts_t * yyg;
	yyg = (struct yyguts_t*)yyscanner;

	//FILE	*TexOpen();

	if (!fFollow)
	    return;
	rgfp[cfp++] = yyin;
	if ((yyin = TexOpen(sbFile)) == NULL) {
	    Warning("can't open \\input file", sbFile);
	    yyin = rgfp[--cfp];
            return;
	} 
#ifdef FLEX_SCANNER
        rgsb[csb++] = YY_CURRENT_BUFFER;
        yy_switch_to_buffer(yy_create_buffer( yyin, YY_BUF_SIZE, yyscanner ), yyscanner );
#endif /* FLEX_SCANNER */
}

/******
** IncludeFile -- if sbFile is not in the rgsbIncList, push current yyin
**	and open sbFile.  If the open fails, the sbFile is ignored.
******/

void
IncludeFile(char *sbFile, yyscan_t yyscanner)
{
	struct yyguts_t * yyg;
	yyg = (struct yyguts_t*)yyscanner;

	//FILE	*TexOpen();

	if (!fFollow)
	    return;
	if (!InList(sbFile))
	    return;
	rgfp[cfp++] = yyin;
	if ((yyin = TexOpen(sbFile)) == NULL) {
	    Warning("can't open \\include file", sbFile);
	    yyin = rgfp[--cfp];
            return;
	}
#ifdef FLEX_SCANNER
        rgsb[csb++] = YY_CURRENT_BUFFER;
        yy_switch_to_buffer(yy_create_buffer( yyin, YY_BUF_SIZE, yyscanner ), yyscanner );
#endif /* FLEX_SCANNER */
}

/******
** AddInclude -- adds sbFile to the rgsbIncList and increments csbIncList.
**	If the include list is too long, sbFile is ignored.
******/

void
AddInclude(char *sbFile)
{
	if (!fFollow)
	    return;
	if (csbIncList >= MAXINCLIST)
	    Warning("\\includeonly list is too long, ignoring", sbFile);
	rgsbIncList[csbIncList] = SafeMalloc(strlen(sbFile) + 1, "malloc for AddInclude failed");
	(void)strcpy(rgsbIncList[csbIncList++], sbFile);
}

/******
** InList -- checks to see if sbFile is in the rgsbIncList.  If there is
**	no list, all files are assumed to be "in the list".
******/

int
InList(char *sbFile)
{
	char	*pch, sbBase[PATH_MAX];
	int	i;

	if (csbIncList == 0)	/* no list */
	    return(1);
	(void)strcpy(sbBase, sbFile);
	if ((pch = strrchr(sbBase, '.')) != NULL)
	    *pch = '\0';
	i = 0;
	while ((i < csbIncList) && rgsbIncList[i])
	    if (strcmp(rgsbIncList[i++], sbBase) == 0)
	        return(1);
	return(0);
}

#ifndef KPATHSEA
/******
** SetInputPaths -- sets rgsbInputPaths to the values indicated by the
**	TEXINPUTS environment variable if set or else DEFAULTINPUTS.  If
**	the user's TEXINPUTS has a leading ':' prepend the DEFAULTINPUTS
**	to the path, if there is a trailing ':' append the DEFAULTINPUTS.
**	This is consistent with the most recent TeX.  However, this
**	routine does not honor the '//' construct (expand subdirs).
******/

void
SetInputPaths(void)
{
	const char *sb;
	char *sbPaths;
	int cchDefaults, cchPaths;

	cchDefaults = strlen(DEFAULTINPUTS);
//#ifdef OS2
//	if ((sb = getenv("TEXINPUT")) == NULL)
//#endif
//	    if ((sb = getenv("TEXINPUTS")) == NULL)
		sb = DEFAULTINPUTS;
	cchPaths = strlen(sb);
	if (sb[0] == CHPATHSEP)
	    cchPaths += cchDefaults;
	if (sb[strlen(sb) - 1] == CHPATHSEP)
	    cchPaths += cchDefaults;
	sbPaths = SafeMalloc(cchPaths + 1, "malloc for SetInputPaths failed");
	sbPaths[0] = '\0';
	if (sb[0] == CHPATHSEP)
	    (void)strcat(sbPaths, DEFAULTINPUTS);
	(void)strcat(sbPaths, sb);
	if (sb[strlen(sb) - 1] == CHPATHSEP)
	    (void)strcat(sbPaths, DEFAULTINPUTS);

	csbInputPaths = SeparateList(sbPaths, rgsbInputPaths, CHPATHSEP, MAXINPUTPATHS);
	if (csbInputPaths == my_ERROR)
#ifdef OS2
	    ErrorExit("TEXINPUT(S) environment variable has too many paths");
#else
	    ErrorExit("TEXINPUTS environment variable has too many paths");
#endif
}
#endif

/******
** SeparateList -- takes a chSep separated list sbList, replaces the
**	chSep's with NULLs and sets rgsbList[i] to the beginning of
**	the ith word in sbList.  The number of words is returned.  A
**	my_ERROR is returned if there are more than csbMax words.
******/

int
SeparateList(char *sbList, char *rgsbList[], char chSep, int csbMax)
{
	int	csbList = 0;

	while (sbList && *sbList && csbList < csbMax) {
	    rgsbList[csbList++] = sbList;
	    if ((sbList = strchr(sbList, chSep))) {
			*sbList++ = '\0';
		}
	}
	return(sbList && *sbList ? my_ERROR : csbList);
}

/******
** TexOpen -- tries to open sbFile in each of the rgsbInputPaths in turn.
**	For each input path the following order is used:
**		file.tex - must be as named, if not there go to the next path
**		file.ext - random extension, try it
**		file     - base name, add .tex and try it
**		file     - try it as is
**	Notice that if file exists in the first path and file.tex exists in
**	one of the other paths, file in the first path is what is opened.
**	If the sbFile begins with a '/', no paths are searched.
******/

FILE *
TexOpen(char *sbFile)
{
	char	*sbNew;
#ifndef KPATHSEA
	char	*pch;
	FILE	*fp;
	int	iPath;
	static char	sbFullPath[PATH_MAX];

	// This feels a bit hacky, but it should work...
	// Trick from: https://stackoverflow.com/questions/42876210/c-fopen-opening-directories
	(void)sprintf(sbFullPath, "%s/%s", mainTexDirectory, sbFile);
	if ((fp = fopen(sbFullPath, "r+")) != NULL)
		return(fp);

	for (iPath = 0; iPath < csbInputPaths; iPath++) {
#ifdef OS2
	    if (*sbFile == '/' || *sbFile == '\\' || strchr(sbFile, ':')) {	/* absolute path */
#else
	    if (*sbFile == '/') {	/* absolute path */
#endif
			(void)sprintf(sbFullPath, "%s", sbFile);
			iPath = csbInputPaths;	/* only check once */
	    } else
			(void)sprintf(sbFullPath, "%s/%s", rgsbInputPaths[iPath], sbFile);
#ifdef OS2
	    pch = sbFullPath;
	    while (pch = strchr(pch, '\\'))
		*pch = '/';
#endif

	    /* If sbFile ends in .tex then it must be there */
	    if ((pch = strrchr(sbFullPath, '.')) != NULL && (strcmp(pch, ".tex") == 0)) {
			if ((fp = fopen(sbFullPath, "r+")) != NULL)
				return(fp);
			else
				continue;
	    }

	    /* if .<ext> then try to open it.  the '.' represents   */
	    /* the beginning of an extension if it is not the first */
	    /* character and it does not follow a '.' or a '/'      */
	    if (pch != NULL && pch > &(sbFullPath[0])
                    && *(pch - 1) != '.' && *(pch - 1) != '/'
		    && (fp = fopen(sbFullPath, "r+")) != NULL) {
			return(fp);
		}

	    /* just base name, add .tex to the name */
	    sbNew = SafeMalloc(strlen(sbFullPath) + 5, "malloc for TexOpen failed");
	    (void)strcpy(sbNew, sbFullPath);
	    (void)strcat(sbNew, ".tex");
	    if ((fp = fopen(sbNew, "r+")) != NULL) {
			free(sbNew);
			return(fp);
		}
		free(sbNew);

	    /* try sbFile regardless */
	    if ((fp = fopen(sbFullPath, "r+")) != NULL)
			return(fp);
	}
	return NULL;
#else
	sbNew = kpse_find_file (sbFile, kpse_tex_format, false);

	if (sbNew == NULL)
	    return NULL;

	return fopen (sbNew, "r+");
#endif
}

/******
** SafeMalloc -- wrapper around malloc() to check for failure.
******/

char *
SafeMalloc(int cch, const char *sbMessage)
{
	char *sb;

	if ((sb = (char *)malloc((unsigned)cch)) == NULL)
	    ErrorExit(sbMessage);
	return(sb);
}

/******
** Warning -- print a warning message preceded by the program name.
******/

void
Warning(const char *sb1, const char *sb2)
{
	(void)fprintf(stderr, "%s: warning: %s %s\n", sbProgName, sb1, sb2);
}

/******
** ErrorExit -- print an error message preceded by the program name.
**	Stdout is flushed and detex exits.
******/

void
ErrorExit(const char *sb1)
{
	(void)fflush(stdout);
	(void)fprintf(stderr, "%s: error: %s\n", sbProgName, sb1);
	exit(1);
}

/******
** UsageExit -- print OS/2 usage message and exit.
******/

void
UsageExit(void)
{
	(void)printf("\n%s [ -clnstw ] [ -e environment-list ] [ filename[.tex] ... ]\n",
		sbProgName);
	puts("  -c  echo LaTeX \\cite, \\ref, and \\pageref values\n  \
-e  <env-list> list of LaTeX environments to ignore\n  \
-l  force latex mode\n  \
-n  do not follow \\input and \\include\n  \
-s  replace control sequences with space\n  \
-t  force tex mode\n  \
-w  word only output");
	exit(0);
}





///////////////////////////////////
// Python module functions
///////////////////////////////////


// Module’s method table
static PyMethodDef DetexPyMethods[] = {
	{"detex", (PyCFunction)detex_run, METH_VARARGS|METH_KEYWORDS, "Run detex on input file."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// Module definition structure
static struct PyModuleDef detexpymodule = {
    PyModuleDef_HEAD_INIT,
    "detexpy",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    DetexPyMethods
};

// Module’s initialization function
PyMODINIT_FUNC PyInit_detexpy(void) {
	PyObject *module = NULL;

	module = PyModule_Create(&detexpymodule);

	PyModule_AddStringConstant(module, "defaultenvstr", DEFAULTENV);
	PyModule_AddObject(module, "defaultenv", PyUnicode_Split(PyUnicode_FromString(DEFAULTENV), PyUnicode_FromString(","), -1));

	return module;
}

// main() method deals with importing the module into Python?
int main(int argc, char *argv[]) {
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("detexpy", PyInit_detexpy);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyImport_ImportModule("detexpy");

    PyMem_RawFree(program);
    return 0;
}






