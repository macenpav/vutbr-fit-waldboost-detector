% ----------------------------------------------------------------------------
% FIT thesis
% ----------------------------------------------------------------------------
% (C) 2007 Zdenek Vasicek, 2008 Michal Bidlo
% vim: set syntax=tex encoding=latin2:
% ----------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{fitthesis}[2007/04/13 Sablona pro BP/DP]

%Vychozi hodnoty parametru
\newif\ifczech \let\ifczech\iftrue
\newif\ifcover \let\ifcover\iffalse
\newif\ifPrint \let\ifPrint\iffalse
\newif\ifWis   \let\ifWis\iftrue


%-----------------------------------------------------------------------------
%Parametry tridy
%-----------------------------------------------------------------------------
\DeclareOption{english}{\let\ifczech\iffalse}
\DeclareOption{cover}{\let\ifcover\iftrue}
\DeclareOption{print}{\let\ifWis\iffalse \let\ifPrint\iftrue}

\DeclareOption*{}
\ProcessOptions\relax
%Jako vychozi pouzijeme tridu REPORT
\ifx\pdfoutput\undefined % nejedeme pod pdftexem
   \LoadClass[a4paper,oneside,onecolumn,11pt]{report}
\else
   \LoadClass[pdftex,a4paper,oneside,onecolumn,11pt]{report}
\fi

%-----------------------------------------------------------------------------
%Balicky
%-----------------------------------------------------------------------------
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[czech,english]{babel}
\usepackage[latin2]{inputenc} %kodovani
\ifczech
  \main@language{czech}
\else
  \main@language{english}
\fi
\ifx\pdfoutput\undefined  % nejedeme pod pdftexem
  \usepackage{graphics}
  \usepackage{epsfig}
\else % je to pdftex !
  \usepackage[pdftex]{graphicx}
\fi

%-----------------------------------------------------------------------------
%Sazebni obrazec
%-----------------------------------------------------------------------------
\setlength{\hoffset}{-0.7cm}
\setlength{\voffset}{-1.5cm}
\setlength{\textheight}{23.0cm}
\setlength{\textwidth}{15.2cm}

%-----------------------------------------------------------------------------
%Texty
%-----------------------------------------------------------------------------
\def\@vut@CS{Vysoké uèení technické v~Brnì}
\def\@vut@EN{Brno University of Technology}
\def\@fit@CS{Fakulta informaèních technologií}
\def\@fit@EN{Faculty of Information Technology}
\def\@author@CS{Autor práce}
\def\@author@EN{Author}
\def\@supervisor@CS{Vedoucí práce}
\def\@supervisor@EN{Supervisor}
\def\@dep@CS{Ústav}
\def\@dep@EN{Department of}
\def\@abstract@CS{Abstrakt}
\def\@abstract@EN{Abstract}
\def\@keywords@CS{Klíèová slova}
\def\@keywords@EN{Keywords}
\def\@cite@CS{Citation}
\def\@copyright@CS{This work has been created as a school publication at Brno University of Technology, Faculty of Information Technology. 
It is hereby a subject to the copyright laws and its usage without the consent of the author is prohibited, except for cases allowed by the law.}
\def\@tdecl@CS{Declaration}
\def\@tackn@CS{Acknowledgment}

\newif\ifnasel
\def\ifEqString#1#2#3#4{\def\testa{#1}\def\testb{#2}%
  \ifx\testa\testb \def#4{#3} \naseltrue \fi%
}

%Vyber typu projektu
\newcommand{\set@project}[1]{%
 \naselfalse
 \ifEqString{#1}{BP}{Bakaláøská práce}{\@projecttype@CS}%
 \ifEqString{#1}{BP}{Bachelor's thesis}{\@projecttype@EN}%
 \ifEqString{#1}{SP}{Semestrální projekt}{\@projecttype@CS}%
 \ifEqString{#1}{SP}{Term project}{\@projecttype@EN}%
 \ifEqString{#1}{DP}{Master's thesis}{\@projecttype@CS}%
 \ifEqString{#1}{DP}{Master's thesis}{\@projecttype@EN}%
 \ifEqString{#1}{DR}{Disertaèní práce}{\@projecttype@CS}%
 \ifEqString{#1}{DR}{PhD thesis}{\@projecttype@EN}%
 \ifnasel
 \else
   \typeout{Spatny typ projektu! Moznosti: BP, SP, DP, DR}\set@project{BP}
 \fi
}

%Vyber ustavu
\newcommand{\set@department}[1]{%
 \naselfalse
 \ifEqString{#1}{UIFS}{informaèních systémù}{\@department@CS}%
 \ifEqString{#1}{UIFS}{Information Systems}{\@department@EN}%
 \ifEqString{#1}{UPSY}{poèítaèových systémù}{\@department@CS}%
 \ifEqString{#1}{UPSY}{Computer Systems}{\@department@EN}%
 \ifEqString{#1}{UITS}{inteligentních systémù}{\@department@CS}%
 \ifEqString{#1}{UITS}{Intelligent Systems}{\@department@EN}%
 \ifEqString{#1}{UPGM}{poèítaèové grafiky a multimédií}{\@department@CS}%
 \ifEqString{#1}{UPGM}{Computer Graphics and Multimedia}{\@department@EN}%
 \ifnasel
 \else
   \typeout{Spatny ustav! Moznosti: UPSY, UIFS, UITS, UPGM}\set@department{UIFS}
 \fi
}

%Nazev ustavu (CS, EN)
\def\@departmentname@CS{\@dep@CS\ \@department@CS}
\def\@departmentname@EN{\@dep@EN\ \@department@EN}

%Nazev prace (v jazyce prace)
\def\@title{\ifczech\@title@CS\else\@title@EN\fi}

%Jmeno, prijmeni, tituly autora
\def\get@author{\ifx\undefined\@author@tp\else%
\@author@tp~\fi\MakeUppercase{\@author}\ifx\undefined\@author@ta\else,~\@author@ta\fi}
%Jmeno, prijmeni, tituly vedouciho
\def\get@supervisor{\ifx\undefined\@supervisor@tp\else%
\@supervisor@tp~\fi\MakeUppercase{\@supervisor}\ifx\undefined\@supervisor@ta\else,~\@supervisor@ta\fi}

%-----------------------------------------------------------------------------
% Projectinfo - parametrizace sablony
%-----------------------------------------------------------------------------
%Parametry 
\define@key{projinfo}{title.cs}{\global\def\@title@CS{#1}}
\define@key{projinfo}{title.en}{\global\def\@title@EN{#1}}
\define@key{projinfo}{department}{\set@department{#1}}
\define@key{projinfo}{project}{\set@project{#1}}
\define@key{projinfo}{year}{\def\@year{#1}}
\define@key{projinfo}{date}{\def\@date{#1}}
\define@key{projinfo}{author}{\def\@author{#1}}
\define@key{projinfo}{author.title.a}{\def\@author@ta{#1}}
\define@key{projinfo}{author.title.p}{\def\@author@tp{#1}}
\define@key{projinfo}{supervisor}{\def\@supervisor{#1}}
\define@key{projinfo}{supervisor.title.a}{\def\@supervisor@ta{#1}}
\define@key{projinfo}{supervisor.title.p}{\def\@supervisor@tp{#1}}
\define@key{projinfo}{keywords.cs}{\def\@tkeywords@CS{#1}}
\define@key{projinfo}{keywords.en}{\def\@tkeywords@EN{#1}}
\define@key{projinfo}{abstract.cs}{\def\@tabstract@CS{#1}}
\define@key{projinfo}{abstract.en}{\def\@tabstract@EN{#1}}
\define@key{projinfo}{declaration}{\def\@declaration{#1}}
\define@key{projinfo}{acknowledgment}{\def\@acknowledgment{#1}}

%Vychozi hodnoty
\setkeys{projinfo}{
  title.cs={Název práce (èesky)},title.en={Název práce(anglicky)}, 
  date=\today, project=BP, department=UIFS, declaration=, 
  abstract.cs=, abstract.en=, keywords.cs=, keywords.en=,
  author=Jmeno Prijmeni, supervisor=Jmeno Prijmeni
}%

\def\projectinfo{\setkeys{projinfo}}

%Prohlaseni
\newcommand{\declaration}[1]{\def\@declaration{#1}}

%Podekovani
\newcommand{\acknowledgment}[1]{\def\@acknowledgment{#1}}

%Abstrakt
\renewcommand{\abstract}[2][cs]{%
 \ifEqString{#1}{cs}{#2}{\@tabstract@CS}%
 \ifEqString{#1}{en}{#2}{\@tabstract@EN}%
}

%Klicova slova
\newcommand{\keywords}[2][cs]{%
 \ifEqString{#1}{cs}{#2}{\@tkeywords@CS}%
 \ifEqString{#1}{en}{#2}{\@tkeywords@EN}%
}

\newlength{\mkspc}
\setlength{\mkspc}{2pt}

\def\setfontsize#1{\renewcommand{\baselinestretch}{1.5}\fontsize{#1}{\f@baselineskip}\selectfont}
%text
\newcommand{\mktxt}[2]{\vbox{\setfontsize{#1}\noindent#2}\vspace\mkspc}
%bold
\newcommand{\mktxt@b}[2]{\vbox{\setfontsize{#1}\noindent\bf#2}\vspace\mkspc}
\newcommand{\mkuctxt}[2]{\vbox{\setfontsize{#1}\noindent\MakeUppercase{#2}}\vspace\mkspc}
%-----------------------------------------------------------------------------
%Desky \makecover
%-----------------------------------------------------------------------------
\newcommand\makecover{\thispagestyle{empty}\null%
%===================================================
\fontfamily{phv}\selectfont%
%  
\mkuctxt{24}{\@vut@CS}
\mkuctxt{11}{\@vut@EN}

\vspace{12mm}

\mkuctxt{14}{\@fit@CS}
\mkuctxt{14}{\@departmentname@CS}

\vspace{4mm}

\mkuctxt{11}{\@fit@EN}
\mkuctxt{11}{\@departmentname@EN}

\vfill

\mkuctxt{18}{\@title} %Nazev prace v jazyce v jakem je napsana

\vfill

\mkuctxt{14}{\@projecttype@CS}
\mkuctxt{10}{\@projecttype@EN}
\vspace{5mm}

\mktxt{14}{\MakeUppercase{\@author@CS} \hfill \get@author}
\mkuctxt{10}{\@author@EN}

\vspace{10mm}

\mktxt{11}{BRNO \@year}

%===================================================
\newpage\fontfamily{\familydefault}\selectfont%
}

%-----------------------------------------------------------------------------
%Abstrakt, klicovaslova \makeabstract
%-----------------------------------------------------------------------------
\newcommand\makeabstract{%
\thispagestyle{empty}\null%

\mktxt@b{16}{\@abstract@CS}\noindent\@tabstract@CS

\vfill

\mktxt@b{16}{\@abstract@EN}\noindent\@tabstract@EN

\vfill

\mktxt@b{16}{\@keywords@CS}\noindent\@tkeywords@CS

\vfill

\mktxt@b{16}{\@keywords@EN}\noindent\@tkeywords@EN

\vfill

\mktxt@b{16}{\@cite@CS}
\noindent\@author: \@title, \MakeLowercase{\@projecttype@CS}, Faculty of Information Technology, BUT, Brno \@year
%===================================================
\newpage
}

%-----------------------------------------------------------------------------
%Prohlaseni, podekovani \makedeclaration
%-----------------------------------------------------------------------------
\newcommand\makedeclaration{
\thispagestyle{empty}%
\mktxt@b{18}{\@title}

\vspace{10mm}
\mktxt@b{16}{\@tdecl@CS}
\noindent\@declaration

  \begin{flushright}
      \makebox[4cm]{\dotfill} \\
      \@author\\
      \@date
  \end{flushright}

\ifx\undefined\@acknowledgment
\else
\vspace{20mm}
\mktxt@b{16}{\@tackn@CS}
\noindent\@acknowledgment
\fi

\vfill

\noindent\copyright\ \@author, \@year.\newline
{\em \@copyright@CS}
%===================================================
\newpage
}

%-----------------------------------------------------------------------------
%Uvodni stranky \maketitle
%-----------------------------------------------------------------------------
\renewcommand\maketitle{%
\ifcover\makecover\fi%
\thispagestyle{empty}\null%
%=====================================================
\fontfamily{phv}\selectfont%
\newsavebox{\@obrlogo}\newsavebox{\@rtext}\newlength{\@vyskartext}\newlength{\@vyskaobr}%

\savebox{\@obrlogo}{\includegraphics[width=3.5cm,keepaspectratio]{fig/vut-zp2}}%logo VUT
\savebox{\@rtext}{\vbox{\mkuctxt{18}{\@vut@CS}\mkuctxt{11}{\@vut@EN}}}%text VUT
\settoheight{\@vyskaobr}{\makebox{\usebox{\@obrlogo}}}% vyska loga
\settoheight{\@vyskartext}{\makebox{\usebox{\@rtext}}}% vyska textu
\addtolength{\@vyskaobr}{-\@vyskartext}%odecteme od vysky loga vysku textu
\noindent\usebox{\@obrlogo}\raisebox{\@vyskaobr}{\hspace{10pt}\usebox{\@rtext}}%zvedneme box s textem

\vspace{5mm}

\savebox{\@obrlogo}{\includegraphics[width=3.5cm,keepaspectratio]{fig/fit-zp2}}%logo FIT
\savebox{\@rtext}{%text FIT, USTAV
\vbox{\mkuctxt{14}{\@fit@CS}\mkuctxt{14}{\@departmentname@CS}
\vspace{3mm}
\mkuctxt{11}{\@fit@EN}\mkuctxt{11}{\@departmentname@EN}}}%
\settoheight{\@vyskaobr}{\makebox{\usebox{\@obrlogo}}}% vyska loga
\settoheight{\@vyskartext}{\makebox{\usebox{\@rtext}}}% vyska textu
\addtolength{\@vyskaobr}{-\@vyskartext}%odecteme od vysky loga vysku textu
\noindent\usebox{\@obrlogo}\raisebox{\@vyskaobr}{\hspace{10pt}\usebox{\@rtext}}%zvedneme box s textem

\vfill

\mkuctxt{18}{\@title@CS}
\mkuctxt{10}{\@title@EN}

\vfill

\mkuctxt{14}{\@projecttype@CS}
\mkuctxt{10}{\@projecttype@EN}

\vspace{5mm}

\mktxt{14}{\MakeUppercase{\@author@CS} \hfill \get@author}
\mkuctxt{10}{\@author@EN}

\vspace{5mm}

\mktxt{14}{\MakeUppercase{\@supervisor@CS} \hfill \get@supervisor}
\mkuctxt{10}{\@supervisor@EN}

\vspace{10mm}

\mktxt{11}{BRNO \@year}

\newpage\fontfamily{\familydefault}\selectfont%
%=================================================================================
\makeabstract\makedeclaration
\clearpage
\setcounter{tocdepth}{2}\setcounter{page}{1}
}

%-----------------------------------------------------------------------------
\def\uv#1{\iflanguage{english}{``#1''}%
                              {\quotedblbase #1\textquotedblleft}}%
%\catcode`\"=\active
%\def"{\bgroup
%    \iflanguage{english}%
%        {``\def"{''\egroup}}%
%        {\quotedblbase\def"{\textquotedblleft\egroup}}}

% Now, switch on what is appropriate for czech:

% czech quotation marks
% \bq - begin quotation, \eq - end quotation
\def\bq{\mbox{\kern.1ex\protect\raisebox{-1.3ex}[0pt][0pt]{''}\kern-.1ex}}
\def\eq{\mbox{\kern-.1ex``\kern.1ex}}
%\setlanguage{\czech}

{%                                      % Begin a group for which " is active
\catcode`\"=\active                     % Make " an active character
\catcode`\@=11                          % Make @ an active character
%
%  \csdoublequoteson
%
%       This macro makes " an active character, resets the control sequence
%       \dblqu@te to L (left), and defines \dq as a replacement for ".
%
\gdef\csdoublequoteson{%                % \csdoublequoteson enables "
    \gdef"{\czechquotes}%               % Define " as \czechquotes
    \global\catcode`\"=\active%         % Make " an active character
    \global\chardef\dq=`\"%             % Double-quote char. via \dq
    \global\let\dblqu@te=L%             % Always start with a left double-quote
    }                                   % End of macro
%
%  \bq, \eq
%
%      These macros define default characters for czech left and right
%      double quotes. Czech opening quote is created from two commas with
%      kerning depending on fontdimen four parameter of current font.
%      Better solution should be specially designed character with
%      proper kernings; if you have such characters in fonts
%      (e.g. in DC-fonts), use it instead. (e.g. define
%      macros \bq and \eq e.g. \def\bq{\char"130 }
%      in your document/style file-- not in csquote.sty!)
%      Similar solution is used for czech right quote.
%
%      \cs existence test, stolen from TeXbook exercise 7.7
\def\ifundefined#1{\expandafter\ifx\csname#1\endcsname\relax }%
%
%      another macro to be more efficient in time and space
\global\chardef\f@@r=4
%
\ifundefined{bq}%
\gdef\bq{\kern-.25\fontdimen\f@@r\font,\kern-.8\fontdimen\f@@r\font,%
                \kern-.35\fontdimen\f@@r\font}%
\fi
\ifundefined{eq}%
\gdef\eq{\kern-.35\fontdimen\f@@r\font`\kern-.8\fontdimen\f@@r\font`%
                \kern-.25\fontdimen\f@@r\font}
\fi
%
% Macro \uv for other usage of \bq and \eq.
%
\ifundefined{uv}\iflanguage{czech}%
        \gdef\uv#1{\bq #1\eq}
\fi
%
% \testquotes macro gives warning if citation span this place
%
\gdef\testquotes{\if R\dblqu@te
        \message{Warning: You forgot right double quote!}%
        \let\dblqu@te=L\fi}
%
%  Define the macro that will be executed whenever " is encountered.
%
\gdef\czechquotes{\protect\czechqu@tes}
\gdef\czechqu@tes{%
        %  If the double-quote is the first character in a new paragraph,
        %  make sure it becomes a left double-quote.  This case can be
        %  detected by checking to see if TeX is currently in vertical mode.
        %  If so, the double-quote is at the beginning of the paragraph
        %  (since " hasn't actually generated any horizontal mode tokens
        %  yet, TeX is still in vertical mode).  If the mode is vertical,
        %  set \dblqu@te equal to L.
        %
        \ifinner\else\ifvmode\testquotes\fi\fi%
        %
        %  Now insert the appropriate left or right double-quote.
        %
        %  If \dblqu@te is L, insert an opening quote and set \dblqu@te to R.
        %
        \if L\dblqu@te\bq\global\let\dblqu@te=R%
        %
        %  Otherwise, save the current \spacefactor, insert '', set \dblqu@te
        %  to L, and reset the original \spacefactor.
        %
        \else%
           \let\xxx=\spacefactor%               % Save the \spacefactor
           \eq%                                 % Insert ending quote
           \global\let\dblqu@te=L%              % and reset \dblqu@te
           \spacefactor\xxx%                    % Reset the \spacefactor
        \fi%                                    % End of \if L\dblqu@te...
        }                                       % End of " macro
}                                               % End of group

\gdef\csdoublequotesoff{%
        \catcode`\"=12%                         % Set " back to other
        }
%
% Czech quotes are default
%
\csdoublequoteson




\endinput
%%
%% End of file `fitthesis.cls'.
