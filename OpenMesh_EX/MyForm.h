#pragma once

#include "DotNetUtilities.h"
#include "Mesh/GUA_OM.h"
#include "Mesh/DP.h"
#include <vector>
#include <iostream>
#include <string>
#include "imageloader.h"
#include <Eigen/Sparse>

using namespace Eigen;

Tri_Mesh *mesh;
Tri_Mesh *tPlace;
bool makeTex = false;
int texNum;

std::vector<OMT::FVIter> FVVector;
std::vector<OMT::FIter> FVector;
std::vector<OMT::VIter> P;//outer
std::vector<OMT::VIter> Inner;
OMT::FIter	tmpF, tmpF2 = tmpF;
xform xf;
GLCamera camera;
GLdouble x, y, z;
float fov = 0.7f;
/*----------------------------------------------------------------------------------------------------*/
// 向量OA叉積向量OB。大於零表示從OA到OB為逆時針旋轉。
double cross(OMT::VIter o, OMT::VIter a, OMT::VIter b)
{
	return (tPlace->point(a.handle())[0] - tPlace->point(o.handle())[0]) * (tPlace->point(b.handle())[1] - tPlace->point(o.handle())[1])
		- (tPlace->point(a.handle())[1] - tPlace->point(o.handle())[1]) * (tPlace->point(b.handle())[0] - tPlace->point(o.handle())[0]);
}
double cross(OMT::VHandle o, OMT::VHandle a, OMT::VHandle b)
{
	return (tPlace->point(a)[0] - tPlace->point(o)[0]) * (tPlace->point(b)[1] - tPlace->point(o)[1])
		- (tPlace->point(a)[1] - tPlace->point(o)[1]) * (tPlace->point(b)[0] - tPlace->point(o)[0]);
}

bool compare_position(OMT::VIter a, OMT::VIter b)
{
	return (tPlace->point(a.handle())[0] < tPlace->point(b.handle())[0]) || 
		(tPlace->point(a.handle())[0] == tPlace->point(b.handle())[0] && tPlace->point(a.handle())[1] < tPlace->point(b.handle())[1]);
}
bool compare_position2(OMT::VHandle a, OMT::VHandle b)
{
	return (tPlace->point(a)[0] < tPlace->point(b)[0]) ||
		(tPlace->point(a)[0] == tPlace->point(b)[0] && tPlace->point(a)[1] < tPlace->point(b)[1]);
}

GLdouble length(OMT::VIter a, OMT::VIter b)
{
	return sqrt(pow(tPlace->point(a.handle())[0] - tPlace->point(b.handle())[0], 2)
		+ pow(tPlace->point(a.handle())[1] - tPlace->point(b.handle())[1], 2)
		+ pow(tPlace->point(a.handle())[2] - tPlace->point(b.handle())[2], 2));
}

double angle(Tri_Mesh::VHandle v0, Tri_Mesh::VHandle v1, Tri_Mesh::VHandle v2)
{
	GLdouble a, b, c;
	a = (tPlace->point(v0) - tPlace->point(v1)).length();
	b = (tPlace->point(v1) - tPlace->point(v2)).length();
	c = (tPlace->point(v0) - tPlace->point(v2)).length();

	return acos((a*a + b*b - c*c) / (2 * a * b));
}

std::vector<double> CalInnerWeight(Tri_Mesh::VHandle v0, std::vector<Tri_Mesh::VHandle> AroundVertex, int InnerVertexAmount) 
{
	std::vector<double> InnerWeight;
	for (int i = 0; i < InnerVertexAmount; i++) {
		double B_isub1, Ri;
		double Wi;

		if (i == 0)
			B_isub1 = angle(v0, AroundVertex[AroundVertex.size() - 1], AroundVertex[i]);
		else
			B_isub1 = angle(v0, AroundVertex[i - 1], AroundVertex[i]);

		if (i == AroundVertex.size() - 1)
			Ri = angle(v0, AroundVertex[0], AroundVertex[i]);
		else
			Ri = angle(v0, AroundVertex[i + 1], AroundVertex[i]);

		Wi = (1/tan(B_isub1)) + (1 / tan(Ri));
		InnerWeight.push_back(Wi);
	}
	return InnerWeight;
}
std::vector<double> CalBoundaryWeight(Tri_Mesh::VHandle v0, std::vector<Tri_Mesh::VHandle> AroundVertex, int InnerVertexAmount) 
{
	std::vector<double> BoundaryWeight;
	for (int i = InnerVertexAmount; i < AroundVertex.size(); i++) {
		double B_isub1, Ri;
		double Wi;

		if (i == 0)
			B_isub1 = angle(v0, AroundVertex[AroundVertex.size() - 1], AroundVertex[i]);
		else
			B_isub1 = angle(v0, AroundVertex[i - 1], AroundVertex[i]);

		if (i == AroundVertex.size() - 1)
			Ri = angle(v0, AroundVertex[0], AroundVertex[i]);
		else
			Ri = angle(v0, AroundVertex[i + 1], AroundVertex[i]);

		Wi = (1 / tan(B_isub1)) + (1 / tan(Ri));
		BoundaryWeight.push_back(Wi);
	}
	return BoundaryWeight;
}
double CalWeightSum(std::vector<double> InnerWeight, std::vector<double> BoundaryWeight) {
	double sum = 0;
	for (int i = 0; i < BoundaryWeight.size(); i++)
		sum += BoundaryWeight[i];
	for (int i = 0; i < InnerWeight.size(); i++)
		sum += InnerWeight[i];
	return sum;
}
/*----------------------------------------------------------------------------------------------------*/
GLuint _textureId;
GLuint loadTexture(Tex* image) {
	GLuint textureId;
	glGenTextures(1, &textureId);
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RGB,
		image->width, image->height,
		0,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		image->pixels);
	return textureId;
}

void initRendering() {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
	glEnable(GL_COLOR_MATERIAL);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	Tex* image;
	switch (texNum)
	{		
	case 0:
		image = loadBMP("t1.bmp");		
		break;
	case 1:
		image = loadBMP("t2.bmp");
		break;
	case 2:
		image = loadBMP("t3.bmp");
		break;
	default:
		image = loadBMP("t1.bmp");
		break;
	}
	_textureId = loadTexture(image);
	delete image;
}
/*----------------------------------------------------------------------------------------------------*/

static const Mouse::button physical_to_logical_map[] = {
	Mouse::NONE, Mouse::ROTATE, Mouse::MOVEXY, Mouse::MOVEZ,
	Mouse::MOVEZ, Mouse::MOVEXY, Mouse::MOVEXY, Mouse::MOVEXY,
};
Mouse::button Mouse_State = Mouse::ROTATE;

namespace OpenMesh_EX {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// MyForm 的摘要
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO:  在此加入建構函式程式碼
			//
		}

	protected:
		/// <summary>
		/// 清除任何使用中的資源。
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	private: System::Windows::Forms::ToolStripMenuItem^  fileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  loadModelToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^  openModelDialog;
	private: System::Windows::Forms::SaveFileDialog^  saveModelDialog;
	private: System::Windows::Forms::ToolStripMenuItem^  saveModelToolStripMenuItem;
	private: HKOGLPanel::HKOGLPanelControl^  hkoglPanelControl1;
	private: System::Windows::Forms::GroupBox^  groupBox1;

	private: System::Windows::Forms::OpenFileDialog^  openTexDialog;
	private: System::Windows::Forms::Button^  button2;
	private: System::Windows::Forms::Button^  Texture3;

	private: System::Windows::Forms::Button^  button1;

	protected:

	private:
		/// <summary>
		/// 設計工具所需的變數。
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// 此為設計工具支援所需的方法 - 請勿使用程式碼編輯器修改
		/// 這個方法的內容。
		/// </summary>
		void InitializeComponent(void)
		{
			HKOGLPanel::HKCOGLPanelCameraSetting^  hkcoglPanelCameraSetting2 = (gcnew HKOGLPanel::HKCOGLPanelCameraSetting());
			HKOGLPanel::HKCOGLPanelPixelFormat^  hkcoglPanelPixelFormat2 = (gcnew HKOGLPanel::HKCOGLPanelPixelFormat());
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->loadModelToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->saveModelToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openModelDialog = (gcnew System::Windows::Forms::OpenFileDialog());
			this->saveModelDialog = (gcnew System::Windows::Forms::SaveFileDialog());
			this->hkoglPanelControl1 = (gcnew HKOGLPanel::HKOGLPanelControl());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->Texture3 = (gcnew System::Windows::Forms::Button());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->openTexDialog = (gcnew System::Windows::Forms::OpenFileDialog());
			this->menuStrip1->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->fileToolStripMenuItem });
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Padding = System::Windows::Forms::Padding(8, 2, 0, 2);
			this->menuStrip1->Size = System::Drawing::Size(817, 27);
			this->menuStrip1->TabIndex = 1;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// fileToolStripMenuItem
			// 
			this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				this->loadModelToolStripMenuItem,
					this->saveModelToolStripMenuItem
			});
			this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			this->fileToolStripMenuItem->Size = System::Drawing::Size(45, 23);
			this->fileToolStripMenuItem->Text = L"File";
			// 
			// loadModelToolStripMenuItem
			// 
			this->loadModelToolStripMenuItem->Name = L"loadModelToolStripMenuItem";
			this->loadModelToolStripMenuItem->Size = System::Drawing::Size(168, 26);
			this->loadModelToolStripMenuItem->Text = L"Load Model";
			this->loadModelToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::loadModelToolStripMenuItem_Click);
			// 
			// saveModelToolStripMenuItem
			// 
			this->saveModelToolStripMenuItem->Name = L"saveModelToolStripMenuItem";
			this->saveModelToolStripMenuItem->Size = System::Drawing::Size(168, 26);
			this->saveModelToolStripMenuItem->Text = L"Save Model";
			this->saveModelToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::saveModelToolStripMenuItem_Click);
			// 
			// openModelDialog
			// 
			this->openModelDialog->FileOk += gcnew System::ComponentModel::CancelEventHandler(this, &MyForm::openModelDialog_FileOk);
			// 
			// saveModelDialog
			// 
			this->saveModelDialog->DefaultExt = L"obj";
			this->saveModelDialog->FileOk += gcnew System::ComponentModel::CancelEventHandler(this, &MyForm::saveModelDialog_FileOk);
			// 
			// hkoglPanelControl1
			// 
			hkcoglPanelCameraSetting2->Far = 1000;
			hkcoglPanelCameraSetting2->Fov = 45;
			hkcoglPanelCameraSetting2->Near = -1000;
			hkcoglPanelCameraSetting2->Type = HKOGLPanel::HKCOGLPanelCameraSetting::CAMERATYPE::ORTHOGRAPHIC;
			this->hkoglPanelControl1->Camera_Setting = hkcoglPanelCameraSetting2;
			this->hkoglPanelControl1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->hkoglPanelControl1->Location = System::Drawing::Point(0, 0);
			this->hkoglPanelControl1->Margin = System::Windows::Forms::Padding(4);
			this->hkoglPanelControl1->Name = L"hkoglPanelControl1";
			hkcoglPanelPixelFormat2->Accumu_Buffer_Bits = HKOGLPanel::HKCOGLPanelPixelFormat::PIXELBITS::BITS_0;
			hkcoglPanelPixelFormat2->Alpha_Buffer_Bits = HKOGLPanel::HKCOGLPanelPixelFormat::PIXELBITS::BITS_0;
			hkcoglPanelPixelFormat2->Stencil_Buffer_Bits = HKOGLPanel::HKCOGLPanelPixelFormat::PIXELBITS::BITS_0;
			this->hkoglPanelControl1->Pixel_Format = hkcoglPanelPixelFormat2;
			this->hkoglPanelControl1->Size = System::Drawing::Size(817, 568);
			this->hkoglPanelControl1->TabIndex = 2;
			this->hkoglPanelControl1->Load += gcnew System::EventHandler(this, &MyForm::hkoglPanelControl1_Load);
			this->hkoglPanelControl1->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &MyForm::hkoglPanelControl1_Paint);
			this->hkoglPanelControl1->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &MyForm::hkoglPanelControl1_MouseDown);
			this->hkoglPanelControl1->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &MyForm::hkoglPanelControl1_MouseMove);
			this->hkoglPanelControl1->MouseWheel += gcnew System::Windows::Forms::MouseEventHandler(this, &MyForm::hkoglPanelControl1_MouseWheel);
			// 
			// groupBox1
			// 
			this->groupBox1->Controls->Add(this->Texture3);
			this->groupBox1->Controls->Add(this->button1);
			this->groupBox1->Controls->Add(this->button2);
			this->groupBox1->Location = System::Drawing::Point(686, 12);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(131, 556);
			this->groupBox1->TabIndex = 3;
			this->groupBox1->TabStop = false;
			// 
			// Texture3
			// 
			this->Texture3->Location = System::Drawing::Point(22, 155);
			this->Texture3->Name = L"Texture3";
			this->Texture3->Size = System::Drawing::Size(87, 25);
			this->Texture3->TabIndex = 3;
			this->Texture3->Text = L"Handsome";
			this->Texture3->UseVisualStyleBackColor = true;
			this->Texture3->Click += gcnew System::EventHandler(this, &MyForm::button3_Click);
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(22, 92);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(87, 25);
			this->button1->TabIndex = 2;
			this->button1->Text = L"Texture2";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &MyForm::button1_Click);
			// 
			// button2
			// 
			this->button2->Location = System::Drawing::Point(22, 34);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(87, 25);
			this->button2->TabIndex = 1;
			this->button2->Text = L"Texture1";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &MyForm::button2_Click);
			// 
			// openTexDialog
			// 
			this->openTexDialog->FileName = L"openFileDialog1";
			this->openTexDialog->FileOk += gcnew System::ComponentModel::CancelEventHandler(this, &MyForm::openFileDialog1_FileOk);
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 15);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(817, 568);
			this->Controls->Add(this->menuStrip1);
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->hkoglPanelControl1);
			this->MainMenuStrip = this->menuStrip1;
			this->Margin = System::Windows::Forms::Padding(4);
			this->Name = L"MyForm";
			this->Text = L"OpenMesh_EX";
			this->Load += gcnew System::EventHandler(this, &MyForm::MyForm_Load);
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->groupBox1->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
private: System::Void hkoglPanelControl1_Load(System::Object^  sender, System::EventArgs^  e)
{
	
}
private: System::Void hkoglPanelControl1_Paint(System::Object^  sender, System::Windows::Forms::PaintEventArgs^  e)
{
	glEnable(GL_COLOR_MATERIAL);
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	point center;
	center[0] = 0.0;
	center[1] = 0.0;
	center[2] = 0.0;
	camera.setupGL(xf * center, 1.0);

	glPushMatrix();
	glMatrixMode(GL_MODELVIEW);
	glMultMatrixd((double *)xf);
	if (mesh != NULL)
		mesh->Render_SolidWireframe();
	if (FVector.size() > 0 && !makeTex)
	{
		for (int i = 0; i < FVector.size(); i++)
			fChosen(FVector[i], mesh);
		hkoglPanelControl1->Invalidate();
	}
	if (makeTex)
	{
		initRendering();
		glEnable(GL_TEXTURE_2D);
		//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glBindTexture(GL_TEXTURE_2D, _textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glPolygonOffset(-0.1f, 1.2f);

		for (OMT::FIter f_it = tPlace->faces_begin(); f_it != tPlace->faces_end(); ++f_it)
		{
			glBegin(GL_TRIANGLES);
			for (OMT::FVIter fv_it = tPlace->fv_iter(f_it); fv_it; ++fv_it)
			{
				glTexCoord2f(mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, fv_it)),
					mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, fv_it)));

				glVertex3f(mesh->point(tPlace->property(tPlace->Mapping_ID, fv_it).handle())[0],
					mesh->point(tPlace->property(tPlace->Mapping_ID, fv_it).handle())[1],
					mesh->point(tPlace->property(tPlace->Mapping_ID, fv_it).handle())[2]);
			}
			glEnd();
		}
	}
	glDisable(GL_POLYGON_OFFSET_FILL);
	glDisable(GL_BLEND);
	hkoglPanelControl1->Invalidate();
	glPopMatrix();
}
private: System::Void hkoglPanelControl1_MouseDown(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
{
	if (e->Button == System::Windows::Forms::MouseButtons::Left ||
		e->Button == System::Windows::Forms::MouseButtons::Middle)
	{
		point center;
		Mouse_State = Mouse::NONE;
		center[0] = 0.0;
		center[1] = 0.0;
		center[2] = 0.0;
		camera.mouse(e->X, e->Y, Mouse_State,
			xf * center,
			1.0, xf);
	}
	if (e->Button == System::Windows::Forms::MouseButtons::Right && mesh != NULL)
	{
		GLdouble P[16];
		GLint V[4];
		GLdouble objX, objY, objZ;
		GLfloat  winX, winY, winZ;
		point center;
		Mouse_State = Mouse::NONE;
		center[0] = 0.0;
		center[1] = 0.0;
		center[2] = 0.0;
		camera.mouse(e->X, e->Y, Mouse_State,
			xf * center,
			1.0, xf);
		x = e->X;
		y = hkoglPanelControl1->Height - e->Y;
		glGetDoublev(GL_PROJECTION_MATRIX, P);
		glGetIntegerv(GL_VIEWPORT, V);

		glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);		
		
		gluUnProject(x, y, winZ, xf, P, V, &objX, &objY, &objZ);
		x = objX; y = objY; z = objZ;

		OMT::FIter f_it;
		OMT::FVIter	fv_it;
		GLdouble min = 9999, d = 0, minF = 9999, s = 0, area = 0;
		GLdouble edge[3], point[3][3], line[3], threeArea[3];
		for (f_it = mesh->faces_begin(); f_it != mesh->faces_end(); ++f_it)
		{
			int i = 0;
			for (fv_it = mesh->fv_iter(f_it); fv_it; ++fv_it)
			{
				d = sqrt(pow(mesh->point(fv_it.handle())[0] - x, 2) + pow(mesh->point(fv_it.handle())[1] - y, 2) +
					pow(mesh->point(fv_it.handle())[2] - z, 2));
				if (d < min)
				{
					min = d;
					//tmp = fv_it;
				}
				point[i][0] = mesh->point(fv_it.handle())[0];
				point[i][1] = mesh->point(fv_it.handle())[1];
				point[i][2] = mesh->point(fv_it.handle())[2];
				line[i] = d;
				i++;
			}

			edge[0] = sqrt(pow(point[0][0] - point[1][0], 2) + pow(point[0][1] - point[1][1], 2) + pow(point[0][2] - point[1][2], 2));
			edge[1] = sqrt(pow(point[1][0] - point[2][0], 2) + pow(point[1][1] - point[2][1], 2) + pow(point[1][2] - point[2][2], 2));
			edge[2] = sqrt(pow(point[2][0] - point[0][0], 2) + pow(point[2][1] - point[0][1], 2) + pow(point[2][2] - point[0][2], 2));
			s = (line[0] + line[1] + edge[0]) / 2;
			threeArea[0] = sqrt(s*(s - line[0])*(s - line[1])*(s - edge[0]));
			s = (line[2] + line[1] + edge[1]) / 2;
			threeArea[1] = sqrt(s*(s - line[2])*(s - line[1])*(s - edge[1]));
			s = (line[2] + line[0] + edge[2]) / 2;
			threeArea[2] = sqrt(s*(s - line[2])*(s - line[0])*(s - edge[2]));
			s = (edge[0] + edge[1] + edge[2]) / 2;
			area = sqrt(s*(s - edge[0])*(s - edge[1])*(s - edge[2]));
			if (abs(area - threeArea[0] - threeArea[1] - threeArea[2]) < minF)
			{
				minF = abs(area - threeArea[0] - threeArea[1] - threeArea[2]);
				tmpF = f_it;
			}
		}

		OMT::FIter* b = new OMT::FIter(tmpF);
		if (FVector.size() == 0)
			FVector.push_back(*b);
		else
		{
			bool jump = false;
			for (int i = 0; i < FVector.size(); i++)
				if (tmpF.handle().idx() == FVector[i].handle().idx())
				{
					jump = true;
					FVector.erase(FVector.begin() + i);
					break;
				}
			for (int i = 0; i < FVector.size() && !jump; i++)
			{
				for (OMT::FFIter ff_it = mesh->ff_iter(FVector[i]); ff_it; ff_it++)
				{
					if (ff_it.handle().idx() == tmpF.handle().idx())
					{
						FVector.push_back(*b);
						jump = true;
						break;
					}
				}
			}
			if (!jump)
			{
				while (FVector.size() > 0)
					FVector.pop_back();
				FVector.push_back(*b);
			}
		}
		makeTex = false;

		hkoglPanelControl1->Invalidate();
	}
}
private: System::Void hkoglPanelControl1_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
{
	if (e->Button == System::Windows::Forms::MouseButtons::Left)
	{
		point center;
		Mouse_State = Mouse::ROTATE;
		center[0] = 0.0;
		center[1] = 0.0;
		center[2] = 0.0;
		camera.mouse(e->X, e->Y, Mouse_State,
			xf * center,
			1.0, xf);
		hkoglPanelControl1->Invalidate();
	}

	if (e->Button == System::Windows::Forms::MouseButtons::Middle)
	{
		point center;
		Mouse_State = Mouse::MOVEXY;
		center[0] = 0.0;
		center[1] = 0.0;
		center[2] = 0.0;
		camera.mouse(e->X, e->Y, Mouse_State,
			xf * center,
			1.0, xf);
		hkoglPanelControl1->Invalidate();
	}

	if (e->Button == System::Windows::Forms::MouseButtons::Right && mesh != NULL)
	{
		GLdouble P[16];
		GLint V[4];
		GLdouble objX, objY, objZ;
		GLfloat  winX, winY, winZ;
		point center;
		Mouse_State = Mouse::NONE;
		center[0] = 0.0;
		center[1] = 0.0;
		center[2] = 0.0;
		camera.mouse(e->X, e->Y, Mouse_State,
			xf * center,
			1.0, xf);
		x = e->X;
		y = hkoglPanelControl1->Height - e->Y;
		glGetDoublev(GL_PROJECTION_MATRIX, P);
		glGetIntegerv(GL_VIEWPORT, V);

		glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);

		gluUnProject(x, y, winZ, xf, P, V, &objX, &objY, &objZ);
		x = objX; y = objY; z = objZ;

		OMT::FIter f_it;
		OMT::FVIter	fv_it;
		GLdouble min = 9999, d = 0, minF = 9999, s = 0, area = 0;
		GLdouble edge[3], point[3][3], line[3], threeArea[3];
		for (f_it = mesh->faces_begin(); f_it != mesh->faces_end(); ++f_it)
		{
			int i = 0;
			for (fv_it = mesh->fv_iter(f_it); fv_it; ++fv_it)
			{
				d = sqrt(pow(mesh->point(fv_it.handle())[0] - x, 2) + pow(mesh->point(fv_it.handle())[1] - y, 2) +
					pow(mesh->point(fv_it.handle())[2] - z, 2));
				if (d < min)
				{
					min = d;
					//tmp = fv_it;
				}
				point[i][0] = mesh->point(fv_it.handle())[0];
				point[i][1] = mesh->point(fv_it.handle())[1];
				point[i][2] = mesh->point(fv_it.handle())[2];
				line[i] = d;
				i++;
			}

			edge[0] = sqrt(pow(point[0][0] - point[1][0], 2) + pow(point[0][1] - point[1][1], 2) + pow(point[0][2] - point[1][2], 2));
			edge[1] = sqrt(pow(point[1][0] - point[2][0], 2) + pow(point[1][1] - point[2][1], 2) + pow(point[1][2] - point[2][2], 2));
			edge[2] = sqrt(pow(point[2][0] - point[0][0], 2) + pow(point[2][1] - point[0][1], 2) + pow(point[2][2] - point[0][2], 2));
			s = (line[0] + line[1] + edge[0]) / 2;
			threeArea[0] = sqrt(s*(s - line[0])*(s - line[1])*(s - edge[0]));
			s = (line[2] + line[1] + edge[1]) / 2;
			threeArea[1] = sqrt(s*(s - line[2])*(s - line[1])*(s - edge[1]));
			s = (line[2] + line[0] + edge[2]) / 2;
			threeArea[2] = sqrt(s*(s - line[2])*(s - line[0])*(s - edge[2]));
			s = (edge[0] + edge[1] + edge[2]) / 2;
			area = sqrt(s*(s - edge[0])*(s - edge[1])*(s - edge[2]));
			if (abs(area - threeArea[0] - threeArea[1] - threeArea[2]) < minF)
			{
				minF = abs(area - threeArea[0] - threeArea[1] - threeArea[2]);
				tmpF = f_it;
			}
		}

		OMT::FIter* b = new OMT::FIter(tmpF);
		if (FVector.size() == 0)
			FVector.push_back(*b);
		else
		{
			bool jump = false;
			for (int i = 0; i < FVector.size(); i++)
				if (tmpF.handle().idx() == FVector[i].handle().idx())
				{
					jump = true;
					break;
				}
			for (int i = 0; i < FVector.size() && !jump; i++)
			{
				for (OMT::FFIter ff_it = mesh->ff_iter(FVector[i]); ff_it; ff_it++)
				{
					if (ff_it.handle().idx() == tmpF.handle().idx())
					{
						FVector.push_back(*b);
						jump = true;
						break;
					}
				}
			}
			if (!jump)
			{

				FVector.push_back(*b);
			}
		}


		hkoglPanelControl1->Invalidate();
	}
}
private: System::Void hkoglPanelControl1_MouseWheel(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
{
	if (e->Delta < 0)
	{
		point center;
		Mouse_State = Mouse::WHEELUP;
		center[0] = 0.0;
		center[1] = 0.0;
		center[2] = 0.0;
		camera.mouse(e->X, e->Y, Mouse_State,
			xf * center,
			1.0, xf);
		hkoglPanelControl1->Invalidate();
	}
	else
	{
		point center;
		Mouse_State = Mouse::WHEELDOWN;
		center[0] = 0.0;
		center[1] = 0.0;
		center[2] = 0.0;
		camera.mouse(e->X, e->Y, Mouse_State,
			xf * center,
			1.0, xf);
		hkoglPanelControl1->Invalidate();
	}
}
private: System::Void loadModelToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	openModelDialog->Filter = "Model(*.obj)|*obj";
	openModelDialog->Multiselect = false;
	openModelDialog->ShowDialog();
}
private: System::Void openModelDialog_FileOk(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e)
{
	std::string filename;
	MarshalString(openModelDialog->FileName, filename);

	if (mesh != NULL)
		delete mesh;

	mesh = new Tri_Mesh;

	if (ReadFile(filename, mesh))
	{
		std::cout << filename << std::endl;
		while (FVector.size() > 0)
			FVector.pop_back();
	}
	hkoglPanelControl1->Invalidate();
}
private: System::Void saveModelToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e)
{
	saveModelDialog->Filter = "Model(*.obj)|*obj";
	saveModelDialog->ShowDialog();
}
private: System::Void saveModelDialog_FileOk(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e)
{
	std::string filename;
	MarshalString(saveModelDialog->FileName, filename);

	if (SaveFile(filename, mesh))
		std::cout << filename << std::endl;
}
private: System::Void MyForm_Load(System::Object^  sender, System::EventArgs^  e) {
}
private: System::Void openFileDialog1_FileOk(System::Object^  sender, System::ComponentModel::CancelEventArgs^  e) {
}
private: System::Void button2_Click(System::Object^  sender, System::EventArgs^  e) {
	CalTextureCoord();
	texNum = 0;
	
	makeTex = true;
	hkoglPanelControl1->Invalidate();
}
void CalTextureCoord()
{
	if (tPlace != NULL)
		delete tPlace;
	tPlace = new Tri_Mesh;

	std::vector<OMP::VHandle> face_vhandles;



	int j = 0, size = 0;
	bool find = false;
	P.clear();
	Inner.clear();
	for (int i = 0; i < FVector.size(); i++)//建立新的mesh
	{
		face_vhandles.clear();
		for (OMT::FVIter fv_it = mesh->fv_iter(FVector[i]); fv_it; ++fv_it)
		{
			for (OMT::VIter v_itr = tPlace->vertices_begin(); v_itr != tPlace->vertices_end(); ++v_itr)
			{
				find = false;
				if (tPlace->point(v_itr) == mesh->point(fv_it.handle()))
				{
					find = true;
					face_vhandles.push_back(tPlace->vertex_handle(v_itr.handle().idx()));
					break;
				}
			}
			if (!find)
			{
				face_vhandles.push_back(tPlace->add_vertex(mesh->point(fv_it.handle())));
				for (OMT::VIter v_itr = tPlace->vertices_begin(); v_itr != tPlace->vertices_end(); ++v_itr)
				{
					if (tPlace->point(v_itr) == mesh->point(fv_it.handle()))
					{
						tPlace->property(tPlace->Mapping_ID, v_itr) = fv_it;
						break;
					}
				}
			}
		}
		tPlace->add_face(face_vhandles);
	}

	for (OMT::VIter v_it = tPlace->vertices_begin(); v_it != tPlace->vertices_end(); ++v_it)//跑過每個點
	{
		if (tPlace->PolyConnectivity::is_boundary(v_it.handle()))//是外部點
		{
			bool repeat = false;
			for (int i = 0; i < P.size(); i++)
				if (v_it.handle().idx() == P[i].handle().idx())
				{
					repeat = true;
					break;
				}
			if (!repeat)
				P.push_back(v_it);
		}

		else//是內部點
		{
			std::vector<OMT::VHandle> vHsAround;
			bool repeat = false;
			for (int i = 0; i < Inner.size(); i++)
				if (v_it.handle().idx() == Inner[i].handle().idx())
				{
					repeat = true;
					break;
				}
			if (!repeat)
			{
				Inner.push_back(v_it);


				for (OMT::VIter v_it2 = tPlace->vertices_begin(); v_it2 != tPlace->vertices_end(); ++v_it2)//所有該內部點周圍的點
					if (v_it2.handle() != v_it.handle())
					{
						for (int i = 0; i < vHsAround.size(); i++)
							if (v_it2.handle() == vHsAround[i])
							{
								repeat = true;
								break;
							}
						if (!repeat)
							vHsAround.push_back(v_it2.handle());
					}

				for (int i = 0; i < vHsAround.size(); i++)//算周圍點對該內部點的重
				{
					GLdouble a, b, c, d, f, angleA, angleB;
					tPlace->property(tPlace->weights, v_it).v.push_back(vHsAround[i]);
					if (i == 0)
					{
						a = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[vHsAround.size() - 1])).length();
						b = (tPlace->point(vHsAround[vHsAround.size() - 1]) - tPlace->point(vHsAround[0])).length();
						c = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[0])).length();
						d = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[i + 1])).length();
						f = (tPlace->point(vHsAround[i + 1]) - tPlace->point(vHsAround[0])).length();

						angleA = acos((a*a + b*b - c*c) / (2 * a * b));
						angleB = acos((d*d + f*f - c*c) / (2 * d * f));
						tPlace->property(tPlace->weights, v_it).w.push_back((1 / tan(angleA)) + (1 / tan(angleB)));
					}
					else if (vHsAround.size() - i == 1)
					{
						a = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[i - 1])).length();
						b = (tPlace->point(vHsAround[i - 1]) - tPlace->point(vHsAround[i])).length();
						c = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[i])).length();
						d = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[0])).length();
						f = (tPlace->point(vHsAround[i]) - tPlace->point(vHsAround[0])).length();

						angleA = acos((a*a + b*b - c*c) / (2 * a * b));
						angleB = acos((d*d + f*f - c*c) / (2 * d * f));
						tPlace->property(tPlace->weights, v_it).w.push_back((1 / tan(angleA)) + (1 / tan(angleB)));
					}
					else
					{
						a = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[i - 1])).length();
						b = (tPlace->point(vHsAround[i - 1]) - tPlace->point(vHsAround[i])).length();
						c = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[i])).length();
						d = (tPlace->point(v_it.handle()) - tPlace->point(vHsAround[i + 1])).length();
						f = (tPlace->point(vHsAround[i]) - tPlace->point(vHsAround[i + 1])).length();

						angleA = acos((a*a + b*b - c*c) / (2 * a * b));
						angleB = acos((d*d + f*f - c*c) / (2 * d * f));
						tPlace->property(tPlace->weights, v_it).w.push_back((1 / tan(angleA)) + (1 / tan(angleB)));
					}
				}
			}
		}
	}
	/*算外部點*/
	swap(P[0], *min_element(P.begin(), P.end(), compare_position));//最左下點移到P[0]
	for (int i = 1; i < P.size(); i++)
		for (int j = i + 1; j < P.size(); j++)
		{
			double c = cross(P[0], P[i], P[j]);
			if (!(c > 0 || (c == 0 && length(P[0], P[i]) < length(P[0], P[j]))))
				swap(P[i], P[j]);
		}

	std::vector<GLdouble> edgeL;
	GLdouble sum = 0, x = 0;

	for (int i = 1; i < P.size(); ++i)
	{
		edgeL.push_back(length(P[i], P[i - 1]));
		sum += edgeL[i - 1];
	}
	edgeL.push_back(length(P[P.size() - 1], P[0]));
	sum += edgeL[P.size() - 1];


	mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, P[0])) = 0;
	mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, P[0])) = 0;
	for (int i = 0; i < edgeL.size() - 1; ++i)
	{
		x += (edgeL[i] / sum) * 4;
		int y = x;
		switch (y)
		{
		case 0:
			mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = x;
			mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = 0;
			break;
		case 1:
			mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = 1;
			mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = x - 1;
			break;
		case 2:
			mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = 3 - x;
			mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = 1;
			break;
		case 3:
			mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = 0;
			mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, P[i + 1])) = 4 - x;
			break;
		}
	}
	/*----------------------------------------------算內部點-------------------------------------------------*/
	if (Inner.size() != 0)
	{
		SparseMatrix<double> A(Inner.size(), Inner.size());

		for (int i = 0; i < Inner.size(); i++)//A填入各內部點的W
		{
			GLdouble W = 0;
			for (int j = 0; j < tPlace->property(tPlace->weights, Inner[i]).w.size(); j++)
				W += tPlace->property(tPlace->weights, Inner[i]).w[j];
			A.insert(i, i) = W;
		}

		for (int i = 0; i < Inner.size(); i++)//填入A剩下的地方 第i列
		{
			for (int j = 0; j < Inner.size(); j++)//其他內部點
				if (i != j)
				{
					double w = 0;
					for (int k = 0; k < tPlace->property(tPlace->weights, Inner[i]).v.size(); ++k)
						if (tPlace->property(tPlace->weights, Inner[i]).v[k].idx() == Inner[j].handle().idx())//k是內部點j
							w = tPlace->property(tPlace->weights, Inner[i]).w[k];
					A.insert(i, j) = -1 * w;
				}
		}

		A.makeCompressed();//A完成

		std::vector<VectorXd> B;
		B.resize(2);
		B[0].resize(Inner.size());
		B[1].resize(Inner.size());

		for (int i = 0; i < Inner.size(); i++)//所有內部點
		{
			B[0][i] = 0;
			B[1][i] = 0;
			for (int j = 0; j < P.size(); j++)//所有外部點
				for (int k = 0; k < tPlace->property(tPlace->weights, Inner[i]).v.size(); k++)//內部點i相連的點
					if (P[j].handle().idx() == tPlace->property(tPlace->weights, Inner[i]).v[k].idx())
					{
						B[0][i] += tPlace->property(tPlace->weights, Inner[i]).w[k] * (mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, tPlace->property(tPlace->weights, Inner[i]).v[k]).handle()));
						B[1][i] += tPlace->property(tPlace->weights, Inner[i]).w[k] * (mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, tPlace->property(tPlace->weights, Inner[i]).v[k]).handle()));
					}
		}
		//B完成

		//解AX = B
		SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> linearSolver;
		linearSolver.compute(A);

		std::vector<VectorXd> X;
		X.resize(2);

		X[0] = linearSolver.solve(B[0]);
		X[1] = linearSolver.solve(B[1]);

		for (int i = 0; i < Inner.size(); i++)
		{
			mesh->property(mesh->coordX, tPlace->property(tPlace->Mapping_ID, Inner[i])) = X[0][i];
			mesh->property(mesh->coordY, tPlace->property(tPlace->Mapping_ID, Inner[i])) = X[1][i];
			std::cout << X[0][i] << "," << X[1][i] << std::endl;
		}

		/*內部參數化完成*/
	}

	std::cout << std::endl;
}
private: System::Void button1_Click(System::Object^  sender, System::EventArgs^  e) {
	CalTextureCoord();
	texNum = 1;

	makeTex = true;
	hkoglPanelControl1->Invalidate();
}
private: System::Void button3_Click(System::Object^  sender, System::EventArgs^  e) {
	CalTextureCoord();
	texNum = 2;

	makeTex = true;
	hkoglPanelControl1->Invalidate();
}
};
}
